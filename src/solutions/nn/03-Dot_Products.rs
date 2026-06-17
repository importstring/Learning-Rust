/*
NN Step 3. Dot Product
[src/solutions/nn/03-DotProduct.rs]

Goal:
Implement the dot product for two 1D vectors stored as Vec<f32>.

Why this matters for neural networks:
Matrix multiplication is built from dot products.
For a dense layer, each output neuron is:

output_j = dot(input_row, weight_column_j) + bias_j

So you want to be very comfortable with computing dot products in code.

Definition:
Given two vectors of the same length:

a = [a0, a1, ..., a_{n-1}]
b = [b0, b1, ..., b_{n-1}]

The dot product is a single number:

dot(a, b) = a0*b0 + a1*b1 + ... + a_{n-1}*b_{n-1}

Shape rules:
- If a.len() != b.len(), return None
- Otherwise, return Some(the dot product as f32)

Examples:

Input:
a = [1.0, 2.0, 3.0]
b = [4.0, 5.0, 6.0]

Output:
Some(32.0)
Explanation:
1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32

Input:
a = [0.0, 0.0]
b = [5.0, 7.0]

Output:
Some(0.0)

Input:
a = [1.5]
b = [2.0]

Output:
Some(3.0)

Input:
a = [1.0, 2.0]
b = [3.0]

Output:
None
Explanation:
Lengths differ (2 vs 1), so the dot product is not defined.
*/

/* Helpful Resources:

Vec iteration and sum:
https://doc.rust-lang.org/std/vec/struct.Vec.html
https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.sum

Dot product intuition:
Dot product is multiply-then-add over matching positions.
Often written as a^T b in linear algebra.
*/

struct Solution;

impl Solution {
    pub fn dot(a: Vec<f32>, b: Vec<f32>) -> Option<f32> {
        if a.len() != b.len() || a.is_empty() {
            return None;
        }

        let mut total = 0.0_f32;
        for i in 0..a.len() {
            total += a[i] * b[i];
        }

        Some(total)
    }
}

// Tests

fn floats_close(x: f32, y: f32, eps: f32) -> bool {
    (x - y).abs() <= eps
}

fn print_result(case_name: &str, a: Vec<f32>, b: Vec<f32>, expected: Option<f32>) {
    let actual = Solution::dot(a.clone(), b.clone());

    let passed = match (actual, expected) {
        (Some(act), Some(exp)) => floats_close(act, exp, 1e-5),
        (None, None) => true,
        _ => false,
    };

    let green = "\x1b[32m";
    let red = "\x1b[31m";
    let cyan = "\x1b[36m";
    let yellow = "\x1b[33m";
    let bold = "\x1b[1m";
    let reset = "\x1b[0m";

    println!("\n{}{}{}{}", bold, cyan, case_name, reset);
    println!("{}Input a:{} {:?}", yellow, reset, a);
    println!("{}Input b:{} {:?}", yellow, reset, b);
    println!("{}Expected:{} {:?}", yellow, reset, expected);
    println!("{}Actual:{}   {:?}", yellow, reset, actual);

    if passed {
        println!("{}✅ PASS{}", green, reset);
    } else {
        println!("{}❌ FAIL{}", red, reset);
    }
}

fn main() {
    println!("\x1b[1m\x1b[35mNN Runner: Dot Product\x1b[0m");

    print_result("Test Case 1", vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0], Some(32.0));

    print_result("Test Case 2", vec![0.0, 0.0], vec![5.0, 7.0], Some(0.0));

    print_result("Test Case 3", vec![1.5], vec![2.0], Some(3.0));

    print_result("Test Case 4", vec![1.0, 2.0], vec![3.0], None);
}