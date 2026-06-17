/*
NN Step 6. ReLU Activation
[src/solutions/nn/06-ReLU.rs]

Goal:
Apply the ReLU activation function elementwise to a 2D matrix.

Why this matters for neural networks:
A dense layer output is usually passed through an activation function
to introduce nonlinearity. One of the most common choices is ReLU:

ReLU(x) = max(0, x)

It:
- keeps positive values unchanged
- sets negative values to 0

Definition (elementwise):
Given a matrix Z with shape (rows x cols),

ReLU(Z)[i][j] = max(0, Z[i][j])

Shape rules:
- The output has the same shape as the input.
- ReLU is applied independently to every element.

Examples:

Input:
Z = [
  [-1.0, 0.0, 2.0],
  [3.0, -4.0, 5.0],
]

Output:
[
  [0.0, 0.0, 2.0],
  [3.0, 0.0, 5.0],
]

Input:
Z = [
  [-0.5],
]

Output:
[
  [0.0],
]

Input:
Z = []
Output:
[]
*/

/* Helpful Resources:

ReLU definition:
f(x) = max(0, x)
https://en.wikipedia.org/wiki/Rectified_linear_unit

*/

struct Solution;

impl Solution {
    pub fn relu(matrix: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        if matrix.is_empty() {
            return vec![];
        }

        let rows = matrix.len();
        let cols = matrix[0].len();

        let mut result = vec![vec![0.0; cols]; rows];

        for (i, row) in matrix.iter().enumerate() {
            for (j, &x) in row.iter().enumerate() {
                result[i][j] = x.max(0.0);
            }
        }

        result
    }
}

// Tests

fn matrices_close(a: &[Vec<f32>], b: &[Vec<f32>], eps: f32) -> bool {
    if a.len() != b.len() {
        return false;
    }

    for i in 0..a.len() {
        if a[i].len() != b[i].len() {
            return false;
        }
        for j in 0..a[i].len() {
            if (a[i][j] - b[i][j]).abs() > eps {
                return false;
            }
        }
    }

    true
}

fn print_result(case_name: &str, matrix: Vec<Vec<f32>>, expected: Vec<Vec<f32>>) {
    let actual = Solution::relu(matrix.clone());
    let passed = matrices_close(&actual, &expected, 1e-5);

    let green = "\x1b[32m";
    let red = "\x1b[31m";
    let cyan = "\x1b[36m";
    let yellow = "\x1b[33m";
    let bold = "\x1b[1m";
    let reset = "\x1b[0m";

    println!("\n{}{}{}{}", bold, cyan, case_name, reset);
    println!("{}Input:{}    {:?}", yellow, reset, matrix);
    println!("{}Expected:{} {:?}", yellow, reset, expected);
    println!("{}Actual:{}   {:?}", yellow, reset, actual);

    if passed {
        println!("{}✅ PASS{}", green, reset);
    } else {
        println!("{}❌ FAIL{}", red, reset);
    }
}

fn main() {
    println!("\x1b[1m\x1b[35mNN Runner: ReLU Activation\x1b[0m");

    print_result(
        "Test Case 1",
        vec![vec![-1.0, 0.0, 2.0], vec![3.0, -4.0, 5.0]],
        vec![vec![0.0, 0.0, 2.0], vec![3.0, 0.0, 5.0]],
    );

    print_result(
        "Test Case 2",
        vec![vec![-0.5]],
        vec![vec![0.0]],
    );

    print_result(
        "Test Case 3",
        vec![vec![1.0, 2.0], vec![3.0, 4.0]],
        vec![vec![1.0, 2.0], vec![3.0, 4.0]],
    );

    print_result(
        "Test Case 4",
        vec![],
        vec![],
    );
}