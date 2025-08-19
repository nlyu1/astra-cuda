#!/usr/bin/env python3
import time
import os
import glob

def read_energy_uj(domain_path):
    """Read energy in microjoules from RAPL interface"""
    try:
        with open(os.path.join(domain_path, 'energy_uj'), 'r') as f:
            return int(f.read().strip())
    except:
        return None

def get_domain_name(domain_path):
    """Get the name of the power domain"""
    try:
        with open(os.path.join(domain_path, 'name'), 'r') as f:
            return f.read().strip()
    except:
        return os.path.basename(domain_path)

def main():
    # Find all RAPL domains
    rapl_base = '/sys/class/powercap/intel-rapl'
    domains = glob.glob(f'{rapl_base}/intel-rapl:*')
    
    if not domains:
        print("No Intel RAPL domains found. This tool requires Intel CPU with RAPL support.")
        return
    
    print("Monitoring power consumption for 60 seconds...")
    print("=" * 60)
    
    # Initial readings
    initial_readings = {}
    for domain in domains:
        name = get_domain_name(domain)
        energy = read_energy_uj(domain)
        if energy is not None:
            initial_readings[domain] = (name, energy)
            # Also check for subdomains
            subdomains = glob.glob(f'{domain}/intel-rapl:*:*')
            for subdomain in subdomains:
                subname = get_domain_name(subdomain)
                subenergy = read_energy_uj(subdomain)
                if subenergy is not None:
                    initial_readings[subdomain] = (f"{name}:{subname}", subenergy)
    
    # Wait 60 seconds
    time.sleep(60)
    
    # Final readings and calculate power
    print("\nPower consumption over 60 seconds:")
    print("-" * 60)
    
    total_package_power = 0
    
    for domain, (name, initial_energy) in initial_readings.items():
        final_energy = read_energy_uj(domain)
        if final_energy is not None:
            # Handle counter wraparound
            if final_energy < initial_energy:
                # Assume 32-bit counter wraparound
                energy_diff = (2**32 - initial_energy) + final_energy
            else:
                energy_diff = final_energy - initial_energy
            
            # Convert microjoules to joules, then to watts (J/s)
            energy_joules = energy_diff / 1_000_000
            power_watts = energy_joules / 60
            
            print(f"{name:20s}: {power_watts:8.2f} W")
            
            # Sum package power (excluding subdomains to avoid double counting)
            if ':' not in os.path.basename(domain) and 'package' in name.lower():
                total_package_power += power_watts
    
    print("-" * 60)
    print(f"{'Total Package Power':20s}: {total_package_power:8.2f} W")
    print("=" * 60)
    
    # Try to read GPU power if available (NVIDIA)
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpu_powers = result.stdout.strip().split('\n')
            print("\nGPU Power:")
            for i, power in enumerate(gpu_powers):
                try:
                    print(f"GPU {i}: {float(power):.2f} W")
                except:
                    pass
    except:
        pass

if __name__ == "__main__":
    main()