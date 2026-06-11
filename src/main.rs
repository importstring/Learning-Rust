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
// class Point: ... ->
//   struct Point { x: f64, y: f64 }
//   impl Point { fn norm(&self) -> f64 { (self.x*self.x + self.y*self.y).sqrt() } }
// s = "hello"; s.upper() ->
//   let s = String::from("hello"); let upper = s.to_uppercase();
// nums[0] -> nums[0]
// for idx, val in enumerate(nums): ->
//   for (idx, val) in nums.iter().enumerate() { ... }
// def is_even(n): return n % 2 == 0 ->
//   fn is_even(n: i32) -> bool { n % 2 == 0 }

struct Solution;

impl Solution {
    pub fn two_sum(nums: Vec<i32>, target: i32) -> Vec<i32> {
        for i in 0..nums.len() { 
            for j in (i+1)..nums.len() {
                if nums[i] + nums[j] == target {
                    return vec![i as i32, j as i32];
                }
            }
        }
        vec![0, 0]
    }
}

// Tests

fn print_result(case_name: &str, nums: Vec<i32>, target: i32, expected: Vec<i32>) {
    let actual = Solution::two_sum(nums.clone(), target);
    let passed = actual == expected;

    let green = "\x1b[32m";
    let red = "\x1b[31m";
    let cyan = "\x1b[36m";
    let yellow = "\x1b[33m";
    let bold = "\x1b[1m";
    let reset = "\x1b[0m";

    println!("\n{}{}{}{}", bold, cyan, case_name, reset);
    println!("{}Input:{} nums = {:?}, target = {}", yellow, reset, nums, target);
    println!("{}Expected:{} {:?}", yellow, reset, expected);
    println!("{}Actual:{}   {:?}", yellow, reset, actual);

    if passed {
        println!("{}✅ PASS{}", green, reset);
    } else {
        println!("{}❌ FAIL{}", red, reset);
    }
}

fn main() {
    println!("\x1b[1m\x1b[35mLeetCode Runner: Two Sum\x1b[0m");

    print_result("Test Case 1", vec![2, 7, 11, 15], 9, vec![0, 1]);
    print_result("Test Case 2", vec![3, 2, 4], 6, vec![1, 2]);
    print_result("Test Case 3", vec![3, 3], 6, vec![0, 1]);
}