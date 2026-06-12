/*
LeetCode 150. Evaluate Reverse Polish Notation
https://leetcode.com/problems/evaluate-reverse-polish-notation/

You are given an array of strings tokens that represents an arithmetic expression
in Reverse Polish Notation.

Evaluate the expression. Return an integer that represents the value of the expression.

Note that:
- The valid operators are '+', '-', '*', and '/'.
- Each operand may be an integer or another expression.
- The division between two integers always truncates toward zero.
- There will not be any division by zero.
- The input represents a valid arithmetic expression in a reverse polish notation.
- The answer and all the intermediate calculations can be represented in a 32-bit integer.
*/

/* Helpful resources:

Stacks and Vec:
- You can use Vec<i32> as a stack:
  - stack.push(x)
  - stack.pop()

String methods:
- token.parse::<i32>()
  https://doc.rust-lang.org/std/primitive.str.html#method.parse

Match expressions and pattern matching:
  https://doc.rust-lang.org/book/ch06-02-match.html
*/

struct Solution;

impl Solution {
    pub fn eval_rpn(tokens: Vec<String>) -> i32 {
        // your implementation goes here
        // Hint:
        // - Create a Vec<i32> as a stack.
        // - Iterate over tokens.
        // - Use `match token.as_str()` to handle "+", "-", "*", "/", or "number".
        // - For operators, pop 2 values, apply the operation in the correct order,
        //   then push the result back.
        // - At the end, the stack should have exactly one value: return it.
    }
}

// Tests

fn print_result(case_name: &str, tokens: Vec<&str>, expected: i32) {
    let tokens_owned: Vec<String> = tokens.iter().map(|s| s.to_string()).collect();
    let actual = Solution::eval_rpn(tokens_owned);
    let passed = actual == expected;

    let green = "\x1b[32m";
    let red = "\x1b[31m";
    let cyan = "\x1b[36m";
    let yellow = "\x1b[33m";
    let bold = "\x1b[1m";
    let reset = "\x1b[0m";

    println!("\n{}{}{}{}", bold, cyan, case_name, reset);
    println!("{}Input:{}   {:?}", yellow, reset, tokens);
    println!("{}Expected:{} {:?}", yellow, reset, expected);
    println!("{}Actual:{}   {:?}", yellow, reset, actual);

    if passed {
        println!("{}✅ PASS{}", green, reset);
    } else {
        println!("{}❌ FAIL{}", red, reset);
    }
}

fn main() {
    println!("\x1b[1m\x1b[35mLeetCode Runner: Evaluate Reverse Polish Notation\x1b[0m");

    // Example 1:
    // Input: ["2","1","+","3","*"]
    // Output: 9
    // Explanation: ((2 + 1) * 3) = 9
    print_result(
        "Test Case 1",
        vec!["2", "1", "+", "3", "*"],
        9,
    );

    // Example 2:
    // Input: ["4","13","5","/","+"]
    // Output: 6
    // Explanation: 4 + (13 / 5) = 6
    print_result(
        "Test Case 2",
        vec!["4", "13", "5", "/", "+"],
        6,
    );

    // Example 3:
    // Input: ["10","6","9","3","+","-11","*","/","*","17","+","5","+"]
    // Output: 22
    print_result(
        "Test Case 3",
        vec!["10","6","9","3","+","-11","*","/","*","17","+","5","+"],
        22,
    );

    // Extra case: single number
    print_result(
        "Test Case 4",
        vec!["42"],
        42,
    );
}