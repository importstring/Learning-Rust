/*
NN Step 5. Matrix + Bias
[src/solutions/nn/05-MatrixPlusBias.rs]

Goal:
Add a 1D bias vector to each row of a 2D matrix.

Why this matters for neural networks:
A dense layer computes:

output = relu(input * weights + bias)

The "+ bias" part is exactly this operation:
- input * weights gives you a matrix of shape (batch_size, out_dim)
- bias has shape (out_dim)
- You add the same bias vector to every row.

Definition:
Given a matrix M with shape (rows x cols) and a bias vector b with length cols:

out[i][j] = M[i][j] + b[j]

Shape rules:
- Let M have shape (rows x cols).
- Let b have length cols.
- If b.len() != cols, the operation is invalid.

Examples:

M = [
  [1.0, 2.0, 3.0],
  [4.0, 5.0, 6.0],
]

b = [10.0, 20.0, 30.0]

out = [
  [11.0, 22.0, 33.0],
  [14.0, 25.0, 36.0],
]

M = [
  [0.0, 0.0],
]

b = [1.0, -1.0]

out = [
  [1.0, -1.0],
]
*/

struct Solution;

impl Solution {
    pub fn add_bias(matrix: Vec<Vec<f32>>, bias: Vec<f32>) -> Option<Vec<Vec<f32>>> {
        if matrix.is_empty() {
            return Some(vec![]);
        }

        let rows = matrix.len();
        let cols = matrix[0].len();

        if bias.len() != cols {
            return None;
        }

        let mut result = vec![vec![0.0; cols]; rows];

        for i in 0..rows {
            for j in 0..cols {
                result[i][j] = matrix[i][j] + bias[j];
            }
        }
        
        Some(result)
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
    bias: Vec<f32>,
    expected: Option<Vec<Vec<f32>>>,
) {
    let actual = Solution::add_bias(matrix.clone(), bias.clone());

    let passed = match (&actual, &expected) {
        (Some(a), Some(e)) => matrices_close(a, e, 1e-5),
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
    println!("{}Matrix:{} {:?}", yellow, reset, matrix);
    println!("{}Bias:{}   {:?}", yellow, reset, bias);
    println!("{}Expected:{} {:?}", yellow, reset, expected);
    println!("{}Actual:{}   {:?}", yellow, reset, actual);

    if passed {
        println!("{}✅ PASS{}", green, reset);
    } else {
        println!("{}❌ FAIL{}", red, reset);
    }
}

fn main() {
    println!("\x1b[1m\x1b[35mNN Runner: Matrix + Bias\x1b[0m");

    print_result(
        "Test Case 1",
        vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
        vec![10.0, 20.0, 30.0],
        Some(vec![vec![11.0, 22.0, 33.0], vec![14.0, 25.0, 36.0]]),
    );

    print_result(
        "Test Case 2",
        vec![vec![0.0, 0.0]],
        vec![1.0, -1.0],
        Some(vec![vec![1.0, -1.0]]),
    );

    print_result(
        "Test Case 3",
        vec![vec![1.0, 2.0]],
        vec![3.0],
        None,
    );

    print_result(
        "Test Case 4",
        vec![],
        vec![],
        Some(vec![]),
    );
}