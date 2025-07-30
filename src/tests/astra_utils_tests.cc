#include "astra_utils.h"
#include <iostream>
#include <vector>

using namespace astra;

void TestStrCat() {
    std::cout << "=== Testing StrCat ===" << std::endl;
    
    // Test basic concatenation
    std::string result1 = StrCat("Hello", " ", "World");
    ASTRA_CHECK_EQ(result1, "Hello World");
    
    // Test with numbers
    std::string result2 = StrCat("Value: ", 42, ", Float: ", 3.14);
    ASTRA_CHECK_EQ(result2, "Value: 42, Float: 3.14");
    
    // Test empty string
    std::string result3 = StrCat("");
    ASTRA_CHECK_EQ(result3, "");
    
    // Test single argument
    std::string result4 = StrCat("single");
    ASTRA_CHECK_EQ(result4, "single");
    
    std::cout << "✓ StrCat tests passed" << std::endl;
}

void TestStrJoin() {
    std::cout << "=== Testing StrJoin ===" << std::endl;
    
    // Test with vector of integers
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    std::string result1 = StrJoin(numbers, ", ");
    ASTRA_CHECK_EQ(result1, "1, 2, 3, 4, 5");
    
    // Test with vector of strings
    std::vector<std::string> words = {"apple", "banana", "cherry"};
    std::string result2 = StrJoin(words, " - ");
    ASTRA_CHECK_EQ(result2, "apple - banana - cherry");
    
    // Test empty vector
    std::vector<int> empty;
    std::string result3 = StrJoin(empty, ", ");
    ASTRA_CHECK_EQ(result3, "");
    
    // Test single element
    std::vector<std::string> single = {"only"};
    std::string result4 = StrJoin(single, ", ");
    ASTRA_CHECK_EQ(result4, "only");
    
    // Test with different delimiter
    std::vector<int> nums = {10, 20, 30};
    std::string result5 = StrJoin(nums, "|");
    ASTRA_CHECK_EQ(result5, "10|20|30");
    
    std::cout << "✓ StrJoin tests passed" << std::endl;
}

void TestStrReplaceAll() {    
    // Test basic replacement
    std::string result1 = StrReplaceAll("hello world hello", "hello", "hi");
    ASTRA_CHECK_EQ(result1, "hi world hi");
    
    // Test no matches
    std::string result2 = StrReplaceAll("hello world", "xyz", "abc");
    ASTRA_CHECK_EQ(result2, "hello world");
    
    // Test empty from string
    std::string result3 = StrReplaceAll("hello", "", "x");
    ASTRA_CHECK_EQ(result3, "hello");
    
    // Test empty input
    std::string result4 = StrReplaceAll("", "a", "b");
    ASTRA_CHECK_EQ(result4, "");
    
    // Test replacement with longer string
    std::string result5 = StrReplaceAll("cat dog cat", "cat", "elephant");
    ASTRA_CHECK_EQ(result5, "elephant dog elephant");
    
    // Test initializer list version
    std::string result6 = StrReplaceAll("hello\nworld\n", {{"\n", "\\n"}});
    ASTRA_CHECK_EQ(result6, "hello\\nworld\\n");
    
    // Test multiple replacements
    std::string result7 = StrReplaceAll("abc def", {{"a", "X"}, {"d", "Y"}});
    ASTRA_CHECK_EQ(result7, "Xbc Yef");
    
    std::cout << "✓ StrReplaceAll tests passed" << std::endl;
}

void TestStrSplit() {
    // Test string delimiter
    std::vector<std::string> result1 = StrSplit("apple,banana,cherry", ",");
    ASTRA_CHECK_EQ(result1.size(), 3);
    ASTRA_CHECK_EQ(result1[0], "apple");
    ASTRA_CHECK_EQ(result1[1], "banana");
    ASTRA_CHECK_EQ(result1[2], "cherry");
    
    // Test char delimiter
    std::vector<std::string> result2 = StrSplit("one two three", ' ');
    ASTRA_CHECK_EQ(result2.size(), 3);
    ASTRA_CHECK_EQ(result2[0], "one");
    ASTRA_CHECK_EQ(result2[1], "two");
    ASTRA_CHECK_EQ(result2[2], "three");
    
    // Test empty string
    std::vector<std::string> result3 = StrSplit("", ",");
    ASTRA_CHECK_EQ(result3.size(), 0);
    
    // Test no delimiters
    std::vector<std::string> result4 = StrSplit("hello", ",");
    ASTRA_CHECK_EQ(result4.size(), 1);
    ASTRA_CHECK_EQ(result4[0], "hello");
    
    // Test consecutive delimiters
    std::vector<std::string> result5 = StrSplit("a,,b", ",");
    ASTRA_CHECK_EQ(result5.size(), 3);
    ASTRA_CHECK_EQ(result5[0], "a");
    ASTRA_CHECK_EQ(result5[1], "");
    ASTRA_CHECK_EQ(result5[2], "b");
    
    std::cout << "✓ StrSplit tests passed" << std::endl;
}

void TestStrSplitFirst() {
    std::cout << "=== Testing StrSplitFirst ===" << std::endl;
    
    // Test basic key=value split
    auto result1 = StrSplitFirst("key=value", "=");
    ASTRA_CHECK_EQ(result1.first, "key");
    ASTRA_CHECK_EQ(result1.second, "value");
    
    // Test with multiple delimiters (only split on first)
    auto result2 = StrSplitFirst("a=b=c=d", "=");
    ASTRA_CHECK_EQ(result2.first, "a");
    ASTRA_CHECK_EQ(result2.second, "b=c=d");
    
    // Test no delimiter found
    auto result3 = StrSplitFirst("no delimiter here", "=");
    ASTRA_CHECK_EQ(result3.first, "no delimiter here");
    ASTRA_CHECK_EQ(result3.second, "");
    
    // Test empty input
    auto result4 = StrSplitFirst("", "=");
    ASTRA_CHECK_EQ(result4.first, "");
    ASTRA_CHECK_EQ(result4.second, "");
    
    // Test delimiter at start
    auto result5 = StrSplitFirst("=value", "=");
    ASTRA_CHECK_EQ(result5.first, "");
    ASTRA_CHECK_EQ(result5.second, "value");
    
    std::cout << "✓ StrSplitFirst tests passed" << std::endl;
}

void TestSimpleAtoi() {
    std::cout << "=== Testing SimpleAtoi ===" << std::endl;
    
    // Test valid integer
    int value;
    bool success = SimpleAtoi("123", &value);
    ASTRA_CHECK_TRUE(success);
    ASTRA_CHECK_EQ(value, 123);
    
    // Test negative integer
    success = SimpleAtoi("-456", &value);
    ASTRA_CHECK_TRUE(success);
    ASTRA_CHECK_EQ(value, -456);
    
    // Test zero
    success = SimpleAtoi("0", &value);
    ASTRA_CHECK_TRUE(success);
    ASTRA_CHECK_EQ(value, 0);
    
    // Test invalid string
    success = SimpleAtoi("abc", &value);
    ASTRA_CHECK_FALSE(success);
    
    // Test partial number
    success = SimpleAtoi("123abc", &value);
    ASTRA_CHECK_FALSE(success);
    
    // Test empty string
    success = SimpleAtoi("", &value);
    ASTRA_CHECK_FALSE(success);
    
    std::cout << "✓ SimpleAtoi tests passed" << std::endl;
}

void TestSimpleAtod() {
    std::cout << "=== Testing SimpleAtod ===" << std::endl;
    
    // Test valid double
    double value;
    bool success = SimpleAtod("3.14", &value);
    ASTRA_CHECK_TRUE(success);
    ASTRA_CHECK_TRUE(value > 3.13 && value < 3.15); // floating point comparison
    
    // Test integer as double
    success = SimpleAtod("42", &value);
    ASTRA_CHECK_TRUE(success);
    ASTRA_CHECK_TRUE(value > 41.9 && value < 42.1);
    
    // Test negative double
    success = SimpleAtod("-2.5", &value);
    ASTRA_CHECK_TRUE(success);
    ASTRA_CHECK_TRUE(value > -2.6 && value < -2.4);
    
    // Test zero
    success = SimpleAtod("0.0", &value);
    ASTRA_CHECK_TRUE(success);
    ASTRA_CHECK_TRUE(value > -0.1 && value < 0.1);
    
    // Test invalid string
    success = SimpleAtod("abc", &value);
    ASTRA_CHECK_FALSE(success);
    
    // Test partial number
    success = SimpleAtod("3.14abc", &value);
    ASTRA_CHECK_FALSE(success);
    
    std::cout << "✓ SimpleAtod tests passed" << std::endl;
}

void TestFormatDouble() {
    std::cout << "=== Testing FormatDouble ===" << std::endl;
    
    // Test integer value
    std::string result1 = FormatDouble(42.0);
    ASTRA_CHECK_EQ(result1, "42.0");
    
    // Test decimal value
    std::string result2 = FormatDouble(3.14159);
    ASTRA_CHECK_TRUE(result2.find("3.14159") == 0); // Should start with 3.14159
    
    // Test zero
    std::string result3 = FormatDouble(0.0);
    ASTRA_CHECK_EQ(result3, "0.0");
    
    // Test negative value
    std::string result4 = FormatDouble(-2.5);
    ASTRA_CHECK_TRUE(result4.find("-2.5") == 0);
    
    std::cout << "✓ FormatDouble tests passed" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "Running Astra Utilities Tests..." << std::endl << std::endl;
    
    TestStrCat();
    TestStrJoin();
    TestStrReplaceAll();
    TestStrSplit();
    TestStrSplitFirst();
    TestSimpleAtoi();
    TestSimpleAtod();
    TestFormatDouble();
    
    std::cout << std::endl << "✓ All utility tests passed!" << std::endl;
    return 0;
} 