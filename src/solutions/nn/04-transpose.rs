/*
NN Step 4. Matrix Transpose
[src/solutions/nn/04-Transpose.rs]

Goal:
Implement the transpose of a 2D matrix stored as Vec<Vec<f32>>.

Why this matters for neural networks:
Transpose is used when:
- Switching between "row-major" and "column-major" views of weights.
- Reusing matmul code in backprop (e.g., W^T * grad).
- Rearranging data for some optimizations.

Definition:
Given a matrix A of shape (rows x cols):

A = [
  [a00, a01, ..., a0(cols-1)],
  [a10, a11, ..., a1(cols-1)],
  ...
  [a(rows-1)0, ..., a(rows-1)(cols-1)]
]

The transpose A^T has shape (cols x rows) and is defined by:

A^T[j][i] = A[i][j] for all valid i, j.

So rows become columns and columns become rows.

Shape rules:
- If A has shape (m x n), the transpose should have shape (n x m).
- An empty matrix (m = 0) can just return an empty matrix.
- You may assume the input is rectangular (all rows same length).

Examples:

Input:
A = [
  [1.0, 2.0, 3.0],
  [4.0, 5.0, 6.0],
]

Output:
[
  [1.0, 4.0],
  [2.0, 5.0],
  [3.0, 6.0],
]

Input:
A = [
  [1.0, 2.0]
]

Output:
[
  [1.0],
  [2.0],
]

Input:
A = [
  [1.0],
  [2.0],
  [3.0],
]

Output:
[
  [1.0, 2.0, 3.0],
]

Input:
A = []
Output:
[]
*/

/* Helpful Resources:

Vec indexing:
https://doc.rust-lang.org/std/vec/struct.Vec.html

Matrix shape intuition:
Transpose swaps rows and columns: A^T[j][i] = A[i][j].
*/

struct Solution;

impl Solution {
    pub fn transpose(matrix: Vec<Vec<f32>>) -> Vec<Vec<f32>> {

        if matrix.is_empty() {
            return vec![];
        }

        let rows = matrix.len();
        let cols = matrix[0].len();
        
        let mut result = vec![vec![0.0; rows]; cols];

        for (idx, row) in matrix.iter().enumerate() {
            for (c_idx, &num) in row.iter().enumerate() {
                result[c_idx][idx] = num;
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

fn print_result(
    case_name: &str,
    matrix: Vec<Vec<f32>>,
    expected: Vec<Vec<f32>>,
) {
    let actual = Solution::transpose(matrix.clone());
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
    println!("\x1b[1m\x1b[35mNN Runner: Matrix Transpose\x1b[0m");

    // 2x3 -> 3x2
    print_result(
        "Test Case 1",
        vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
        vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]],
    );

    // 1x2 -> 2x1
    print_result(
        "Test Case 2",
        vec![vec![1.0, 2.0]],
        vec![vec![1.0], vec![2.0]],
    );

    // 3x1 -> 1x3
    print_result(
        "Test Case 3",
        vec![vec![1.0], vec![2.0], vec![3.0]],
        vec![vec![1.0, 2.0, 3.0]],
    );

    // empty
    print_result(
        "Test Case 4",
        vec![],
        vec![],
    );
}