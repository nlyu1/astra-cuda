#!/usr/bin/env python3
"""
Integration tests for VecMarket CUDA implementation.
Tests the Python bindings and verifies functionality matches the C++ unit tests.
"""

import pytest
import torch
import sys
import os

import astra_cuda
aom = astra_cuda.order_matching


def create_tensors(values, dtype=torch.uint32, device='cuda'):
    """Helper to create CUDA tensors from values."""
    if isinstance(values, int):
        return torch.tensor([values], dtype=dtype, device=device)
    return torch.tensor(values, dtype=dtype, device=device)


class TestVecMarket:
    """Test suite for VecMarket functionality."""
    
    def test_basic_instantiation(self):
        """Test basic market instantiation and constants."""
        market = aom.VecMarket(
            num_markets=10,
            max_price_levels=128,
            max_active_orders_per_market=1024,
            max_active_fills_per_market=512
        )
        
        # Check exposed constants
        assert aom.MAX_MARKETS == 106496
        assert aom.PRICE_LEVELS == 128
        assert aom.NULL_INDEX == 0xFFFFFFFF
    
    def test_partial_fills(self):
        """Test partial order fills."""
        market = aom.VecMarket(1, 128, 1024, 512)
        fills = market.new_fill_batch()
        
        # Add a large buy order
        market.add_two_sided_quotes(
            bid_px=create_tensors(50),
            bid_sz=create_tensors(100),
            ask_px=create_tensors(0),
            ask_sz=create_tensors(0),
            customer_ids=create_tensors(0),
            fills=fills
        )
        
        # Add a smaller sell order that partially fills
        market.add_two_sided_quotes(
            bid_px=create_tensors(0),
            bid_sz=create_tensors(0),
            ask_px=create_tensors(50),
            ask_sz=create_tensors(30),
            customer_ids=create_tensors(1),
            fills=fills
        )
        
        # Check fills
        fill_counts = fills.fill_counts.cpu()
        assert fill_counts[0].item() == 1
        
        fill_sizes = fills.fill_sizes.cpu()
        assert fill_sizes[0][0].item() == 30
        
        # Add another sell order to check remaining buy
        market.add_two_sided_quotes(
            bid_px=create_tensors(0),
            bid_sz=create_tensors(0),
            ask_px=create_tensors(50),
            ask_sz=create_tensors(80),
            customer_ids=create_tensors(2),
            fills=fills
        )
        
        # Should fill only 70 shares (remaining from original 100)
        assert fill_counts[0].item() == 1
        assert fill_sizes[0][0].item() == 70
    
    def test_market_independence(self):
        """Test that markets operate independently."""
        num_markets = 5
        market = aom.VecMarket(num_markets, 128, 1024, 512)
        fills = market.new_fill_batch()
        
        # Set up different scenarios for each market
        bid_prices = create_tensors([10, 20, 30, 40, 50])
        bid_sizes = create_tensors([100, 200, 300, 400, 500])
        ask_prices = create_tensors([15, 25, 35, 45, 55])
        ask_sizes = create_tensors([50, 100, 150, 200, 250])
        customer_ids = create_tensors([0, 0, 0, 0, 0])
        
        market.add_two_sided_quotes(
            bid_px=bid_prices,
            bid_sz=bid_sizes,
            ask_px=ask_prices,
            ask_sz=ask_sizes,
            customer_ids=customer_ids,
            fills=fills
        )
        
        # Verify no fills (no crossing orders)
        fill_counts = fills.fill_counts.cpu()
        for i in range(num_markets):
            assert fill_counts[i].item() == 0
        
        # Add crossing orders to specific markets only
        bid_prices2 = create_tensors([0, 30, 0, 50, 0])
        bid_sizes2 = create_tensors([0, 50, 0, 100, 0])
        ask_prices2 = create_tensors([0, 20, 0, 40, 0])
        ask_sizes2 = create_tensors([0, 75, 0, 150, 0])
        customer_ids2 = create_tensors([1, 1, 1, 1, 1])
        
        market.add_two_sided_quotes(
            bid_px=bid_prices2,
            bid_sz=bid_sizes2,
            ask_px=ask_prices2,
            ask_sz=ask_sizes2,
            customer_ids=customer_ids2,
            fills=fills
        )
        
        # Verify fills only in markets 1 and 3
        fill_counts = fills.fill_counts.cpu()
        assert fill_counts[0].item() == 0  # Market 0: no crossing
        assert fill_counts[1].item() == 1  # Market 1: should have fill
        assert fill_counts[2].item() == 0  # Market 2: no crossing
        assert fill_counts[3].item() == 1  # Market 3: should have fill
        assert fill_counts[4].item() == 0  # Market 4: no crossing
    
    def test_bbo_functionality(self):
        """Test Best Bid/Offer extraction."""
        num_markets = 3
        market = aom.VecMarket(num_markets, 128, 1024, 512)
        fills = market.new_fill_batch()
        
        # Set up orders at different price levels
        bid_prices = create_tensors([10, 15, 20])
        bid_sizes = create_tensors([100, 200, 300])
        ask_prices = create_tensors([25, 30, 35])
        ask_sizes = create_tensors([150, 250, 350])
        customer_ids = create_tensors([0, 0, 0])
        
        market.add_two_sided_quotes(
            bid_px=bid_prices,
            bid_sz=bid_sizes,
            ask_px=ask_prices,
            ask_sz=ask_sizes,
            customer_ids=customer_ids,
            fills=fills
        )
        
        # Get BBOs
        bbo = market.new_bbo_batch()
        market.get_bbos(bbo)
        
        # Check each market's BBO
        best_bid_px = bbo.best_bid_prices.cpu()
        best_bid_sz = bbo.best_bid_sizes.cpu()
        best_ask_px = bbo.best_ask_prices.cpu()
        best_ask_sz = bbo.best_ask_sizes.cpu()
        
        assert best_bid_px[0].item() == 10
        assert best_bid_sz[0].item() == 100
        assert best_ask_px[0].item() == 25
        assert best_ask_sz[0].item() == 150
        
        assert best_bid_px[1].item() == 15
        assert best_ask_px[1].item() == 30
        
        assert best_bid_px[2].item() == 20
        assert best_ask_px[2].item() == 35
    
    def test_zero_size_orders(self):
        """Test that zero-size orders are properly ignored."""
        num_markets = 3
        market = aom.VecMarket(num_markets, 128, 1024, 512)
        fills = market.new_fill_batch()
        
        # Mix of zero and non-zero orders
        bid_prices = create_tensors([10, 20, 30])
        bid_sizes = create_tensors([0, 100, 0])  # Zero sizes
        ask_prices = create_tensors([15, 0, 35])
        ask_sizes = create_tensors([50, 0, 200])  # Zero sizes
        customer_ids = create_tensors([0, 0, 0])
        
        market.add_two_sided_quotes(
            bid_px=bid_prices,
            bid_sz=bid_sizes,
            ask_px=ask_prices,
            ask_sz=ask_sizes,
            customer_ids=customer_ids,
            fills=fills
        )
        
        # Get BBOs to verify zero-size orders were ignored
        bbo = market.new_bbo_batch()
        market.get_bbos(bbo)
        best_bid_px = bbo.best_bid_prices.cpu()
        
        # Market 0: No bid (size was 0)
        assert best_bid_px[0].item() == aom.NULL_INDEX
        
        # Market 1: Has bid
        assert best_bid_px[1].item() == 20
        
        # Market 2: No bid (size was 0)
        assert best_bid_px[2].item() == aom.NULL_INDEX
    
    def test_price_crossing_logic(self):
        """Test various price crossing scenarios."""
        market = aom.VecMarket(1, 128, 1024, 512)
        fills = market.new_fill_batch()
        
        # Test 1: Exact price match
        market.add_two_sided_quotes(
            bid_px=create_tensors(50),
            bid_sz=create_tensors(100),
            ask_px=create_tensors(50),
            ask_sz=create_tensors(100),
            customer_ids=create_tensors(0),
            fills=fills
        )
        
        fill_counts = fills.fill_counts.cpu()
        assert fill_counts[0].item() == 1
        
        # Test 2: Bid higher than ask
        market.add_two_sided_quotes(
            bid_px=create_tensors(60),
            bid_sz=create_tensors(50),
            ask_px=create_tensors(55),
            ask_sz=create_tensors(50),
            customer_ids=create_tensors(1),
            fills=fills
        )
        
        assert fill_counts[0].item() == 1
        
        # Test 3: No cross (bid < ask)
        market.add_two_sided_quotes(
            bid_px=create_tensors(45),
            bid_sz=create_tensors(100),
            ask_px=create_tensors(50),
            ask_sz=create_tensors(100),
            customer_ids=create_tensors(2),
            fills=fills
        )
        
        assert fill_counts[0].item() == 0
    
    def test_execution_price_determination(self):
        """Test that execution happens at the resting order price."""
        market = aom.VecMarket(1, 128, 1024, 512)
        fills = market.new_fill_batch()
        
        # Test 1: Buy order rests, sell order takes
        # Add buy order first (resting)
        market.add_two_sided_quotes(
            bid_px=create_tensors(55),
            bid_sz=create_tensors(100),
            ask_px=create_tensors(0),
            ask_sz=create_tensors(0),
            customer_ids=create_tensors(0),
            fills=fills
        )
        
        # Add sell order (taker) at lower price
        market.add_two_sided_quotes(
            bid_px=create_tensors(0),
            bid_sz=create_tensors(0),
            ask_px=create_tensors(50),
            ask_sz=create_tensors(50),
            customer_ids=create_tensors(1),
            fills=fills
        )
        
        # Check execution price
        fill_prices = fills.fill_prices.cpu()
        assert fill_prices[0][0].item() == 55  # Should execute at bid price (resting)
        
        # Test 2: Sell order rests, buy order takes
        market2 = aom.VecMarket(1, 128, 1024, 512)
        fills2 = market2.new_fill_batch()
        
        # Add sell order first (resting)
        market2.add_two_sided_quotes(
            bid_px=create_tensors(0),
            bid_sz=create_tensors(0),
            ask_px=create_tensors(50),
            ask_sz=create_tensors(100),
            customer_ids=create_tensors(0),
            fills=fills2
        )
        
        # Add buy order (taker) at higher price
        market2.add_two_sided_quotes(
            bid_px=create_tensors(55),
            bid_sz=create_tensors(50),
            ask_px=create_tensors(0),
            ask_sz=create_tensors(0),
            customer_ids=create_tensors(1),
            fills=fills2
        )
        
        # Check execution price
        fill_prices2 = fills2.fill_prices.cpu()
        assert fill_prices2[0][0].item() == 50  # Should execute at ask price (resting)
    
    def test_fill_structure_completeness(self):
        """Test that FillBatch contains all expected fields."""
        market = aom.VecMarket(1, 128, 1024, 512)
        fills = market.new_fill_batch()
        
        # Create a simple crossing trade
        market.add_two_sided_quotes(
            bid_px=create_tensors(50),
            bid_sz=create_tensors(100),
            ask_px=create_tensors(0),
            ask_sz=create_tensors(0),
            customer_ids=create_tensors(5),  # Customer 5
            fills=fills
        )
        
        market.add_two_sided_quotes(
            bid_px=create_tensors(0),
            bid_sz=create_tensors(0),
            ask_px=create_tensors(50),
            ask_sz=create_tensors(60),
            customer_ids=create_tensors(7),  # Customer 7
            fills=fills
        )
        
        # Verify all fill fields are populated correctly
        assert fills.fill_counts.cpu()[0].item() == 1
        assert fills.fill_prices.cpu()[0][0].item() == 50
        assert fills.fill_sizes.cpu()[0][0].item() == 60
        assert fills.fill_customer_ids.cpu()[0][0].item() == 7  # Taker
        assert fills.fill_quoter_ids.cpu()[0][0].item() == 5    # Quoter (resting)
        assert fills.fill_is_sell_quote.cpu()[0][0].item() == False  # Buy was resting
        
        # Verify TID fields exist
        assert hasattr(fills, 'fill_tid')
        assert hasattr(fills, 'fill_quote_tid')
    
    def test_multi_gpu_support(self):
        """Test multi-GPU support if available."""
        if torch.cuda.device_count() < 2:
            pytest.skip("Multi-GPU test requires at least 2 GPUs")
        
        # Create market on GPU 0
        market0 = aom.VecMarket(
            num_markets=100,
            max_price_levels=128,
            max_active_orders_per_market=64,
            max_active_fills_per_market=1024,
            num_customers=16,
            device_id=0,
            threads_per_block=256
        )
        
        # Create market on GPU 1
        market1 = aom.VecMarket(
            num_markets=200,
            max_price_levels=128,
            max_active_orders_per_market=64,
            max_active_fills_per_market=1024,
            num_customers=16,
            device_id=1,
            threads_per_block=512
        )
        
        # Verify they can operate independently
        fills0 = market0.new_fill_batch()
        fills1 = market1.new_fill_batch()
        
        # Add orders to each market
        market0.add_two_sided_quotes(
            bid_px=torch.full((100,), 50, dtype=torch.uint32, device='cuda:0'),
            bid_sz=torch.full((100,), 100, dtype=torch.uint32, device='cuda:0'),
            ask_px=torch.full((100,), 50, dtype=torch.uint32, device='cuda:0'),
            ask_sz=torch.full((100,), 100, dtype=torch.uint32, device='cuda:0'),
            customer_ids=torch.zeros(100, dtype=torch.uint32, device='cuda:0'),
            fills=fills0
        )
        
        market1.add_two_sided_quotes(
            bid_px=torch.full((200,), 60, dtype=torch.uint32, device='cuda:1'),
            bid_sz=torch.full((200,), 200, dtype=torch.uint32, device='cuda:1'),
            ask_px=torch.full((200,), 60, dtype=torch.uint32, device='cuda:1'),
            ask_sz=torch.full((200,), 200, dtype=torch.uint32, device='cuda:1'),
            customer_ids=torch.zeros(200, dtype=torch.uint32, device='cuda:1'),
            fills=fills1
        )
        
        # Verify fills occurred on both GPUs
        assert fills0.fill_counts.cpu()[0].item() == 1
        assert fills1.fill_counts.cpu()[0].item() == 1


def test_string_representation():
    """Test market string representation."""
    market = aom.VecMarket(2, 128, 1024, 512)
    fills = market.new_fill_batch()
    
    # Add some orders
    market.add_two_sided_quotes(
        bid_px=create_tensors([50, 60]),
        bid_sz=create_tensors([100, 200]),
        ask_px=create_tensors([55, 65]),
        ask_sz=create_tensors([150, 250]),
        customer_ids=create_tensors([0, 1]),
        fills=fills
    )
    
    # Get string representation of market 0
    market_str = market.to_string(0)
    assert "Market 0" in market_str
    assert "Bids:" in market_str
    assert "Asks:" in market_str


if __name__ == "__main__":
    # Run tests with pytest if available, otherwise run directly
    try:
        import pytest
        pytest.main([__file__, "-v", "-s"])
    except ImportError:
        print("Running tests without pytest...")
        test_obj = TestVecMarket()
        
        # Run all test methods
        for attr in dir(test_obj):
            if attr.startswith('test_'):
                print(f"\nRunning {attr}...")
                try:
                    getattr(test_obj, attr)()
                    print(f"✓ {attr} passed")
                except Exception as e:
                    print(f"✗ {attr} failed: {e}")
        
        # Run standalone tests
        print("\nRunning test_string_representation...")
        try:
            test_string_representation()
            print("✓ test_string_representation passed")
        except Exception as e:
            print(f"✗ test_string_representation failed: {e}")