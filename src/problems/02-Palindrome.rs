/*
LeetCode 125. Valid Palindrome
https://leetcode.com/problems/valid-palindrome/

Given a string s, return true if it is a palindrome, or false otherwise.

A string is a palindrome if, after converting all uppercase letters to lowercase letters
and removing all non-alphanumeric characters, it reads the same forward and backward.

Examples:

Input:  s = "A man, a plan, a canal: Panama"
Output: true
Explanation: "amanaplanacanalpanama" is a palindrome.

Input:  s = "race a car"
Output: false
Explanation: "raceacar" is not a palindrome.

Input:  s = " "
Output: true
Explanation: s becomes the empty string "", which is a palindrome.
*/

/* Helpful Resources:
String methods:
s.to_lowercase()
https://doc.rust-lang.org/std/index.html?search=to_lowercase

You'll need to know filter:
https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.filter

chars will turn it into an iterable you can use

.collect
*/


struct Solution;

impl Solution {
    pub fn is_palindrome(s: String) -> bool {
        // .. your code goes here
    }
}


// Tests

fn print_result(case_name: &str, s: &str, expected: bool) {
    let actual = Solution::is_palindrome(s.to_string());
    let passed = actual == expected;

    let green = "\x1b[32m";
    let red = "\x1b[31m";
    let cyan = "\x1b[36m";
    let yellow = "\x1b[33m";
    let bold = "\x1b[1m";
    let reset = "\x1b[0m";

    println!("\n{}{}{}{}", bold, cyan, case_name, reset);
    println!("{}Input:{} s = {:?}", yellow, reset, s);
    println!("{}Expected:{} {:?}", yellow, reset, expected);
    println!("{}Actual:{}   {:?}", yellow, reset, actual);

    if passed {
        println!("{}✅ PASS{}", green, reset);
    } else {
        println!("{}❌ FAIL{}", red, reset);
    }
}

fn main() {
    println!("\x1b[1m\x1b[35mLeetCode Runner: Valid Palindrome\x1b[0m");

    print_result("Test Case 1", "A man, a plan, a canal: Panama", true);
    print_result("Test Case 2", "race a car", false);
    print_result("Test Case 3", " ", true);
    print_result("Test Case 4", "0P", false);
}