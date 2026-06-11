// Python → Rust quick cheats
// def add(a, b): -> fn add(a: i32, b: i32) -> i32 { a + b }
// LUCKYNUM = 5 -> let LUCKYNUM: i32 = 5;                  // immutable
// x = 7 -> let mut x: i32 = 7;                           // mutable
// y += x -> y += x;
// print("Hello, World!") -> println!("Hello, World!");
// arr = [1, 2, 3] -> let mut arr: Vec<i32> = Vec::new(); arr.push(1); arr.push(2); arr.push(3);
// other = [4, 5, 6] -> let other = vec![4, 5, 6];
// array.append(3) -> arr.push(3);
// if x > 0: ... elif ... else ... ->
//   if x > 0 { ... } else if x == 0 { ... } else { ... }
// for i in range(5): -> for i in 0..5 { ... }
// for val in arr: -> for val in &arr { ... }
// len(arr) -> arr.len()
// sum(arr) -> arr.iter().sum::<i32>()
// [n*n for n in nums] ->
//   nums.iter().map(|n| n * n).collect::<Vec<_>>()
// dict = {"a": 1} ->
//   use std::collections::HashMap;
//   let mut dict: HashMap<&str, i32> = HashMap::new(); dict.insert("a", 1);
// for k, v in dict.items(): -> for (k, v) in &dict { ... }
// s = "hello"; s.upper() ->
//   let s = String::from("hello"); let upper = s.to_uppercase();
// nums[0] -> nums[0]
// for idx, val in enumerate(nums): ->
//   for (idx, val) in nums.iter().enumerate() { ... }
// def is_even(n): return n % 2 == 0 ->
//   fn is_even(n: i32) -> bool { n % 2 == 0 }


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

struct Solution;

impl Solution {
    pub fn is_palindrome(s: String) -> bool {
        let filtered: String = s
            .to_lowercase()
            .chars() 
            .filter(|c| c.is_alphanumeric())
            .collect();
        
        let reveresed: String = filtered
            .chars()
            .rev()
            .collect();

        filtered == reversed
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