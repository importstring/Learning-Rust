// def add(a, b):
//     return a + b
// -->
fn add(a: i32, b: i32) -> i32 {
    a + b
}

// LUCKYNUM = 5
// -->
let LUCKYNUM: i32 = 5; // immutable variable

// x = 7
// -->
let mut x: i32 = 7; // mutable variable

// y += x
// -->
y += x;

// print("Hello, World!")
// -->
println!("Hello, World!");

// arr = [1, 2, 3]
// -->
let mut arr: Vec<i32> = Vec::new();
arr.push(1);
arr.push(2);
arr.push(3);

// array.append(3)
// -->
arr.push(3);

// other = [4, 5, 6]
// -->
let other = vec![4, 5, 6];

// if x > 0:
//     print("positive")
// elif x == 0:
//     print("zero")
// else:
//     print("negative")
// -->
if x > 0 {
    println!("positive");
} else if x == 0 {
    println!("zero");
} else {
    println!("negative");
}

// for i in range(5):
//     print(i)
// -->
for i in 0..5 {
    println!("{i}");
}

// for val in arr:
//     print(val)
// -->
for val in &arr {
    println!("{val}");
}

// sum(arr)
// -->
let sum: i32 = arr.iter().sum();

// len(arr)
// -->
let length = arr.len();

// def squared(nums):
//     return [n*n for n in nums]
// -->
fn squared(nums: &[i32]) -> Vec<i32> {
    nums.iter().map(|n| n * n).collect()
}

// dict = {"a": 1, "b": 2}
// -->
use std::collections::HashMap;

let mut dict: HashMap<&str, i32> = HashMap::new();
dict.insert("a", 1);
dict.insert("b", 2);

// dict["a"] = 10
// -->
if let Some(value) = dict.get_mut("a") {
    *value = 10;
}

// for key, value in dict.items():
//     print(key, value)
// -->
for (key, value) in &dict {
    println!("{key} {value}");
}

// try:
//     val = int(s)
// except:
//     val = 0
// -->
fn parse_int(s: &str) -> i32 {
    match s.trim().parse::<i32>() {
        Ok(v) => v,
        Err(_) => 0,
    }
}

// class Point:
//     def __init__(self, x, y):
//         self.x = x
//         self.y = y
// -->
struct Point {
    x: f64,
    y: f64,
}

// p = Point(1.0, 2.0)
// -->
let p = Point { x: 1.0, y: 2.0 };

// def norm(self):
//     return (self.x**2 + self.y**2)**0.5
// -->
impl Point {
    fn norm(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
}

// s = "hello"
// s.upper()
// -->
let s = String::from("hello");
let upper = s.to_uppercase();

// nums = [1, 2, 3]
// nums[0]
// -->
let nums = vec![1, 2, 3];
let first = nums[0];

// for idx, val in enumerate(nums):
//     print(idx, val)
// -->
for (idx, val) in nums.iter().enumerate() {
    println!("{idx} {val}");
}

// def is_even(n): return n % 2 == 0
// -->
fn is_even(n: i32) -> bool {
    n % 2 == 0
}

// filtered = [n for n in nums if is_even(n)]
// -->
let filtered: Vec<i32> = nums
    .iter()
    .copied()
    .filter(|n| is_even(*n))
    .collect();

// print(f"value = {x}")
// -->
println!("value = {x}");