/*
NN Step 1. Matrix Multiplication
[src/solutions/nn/01-MatrixMultiplication.rs]

Goal:
Implement matrix multiplication for two 2D matrices stored as Vec<Vec<f32>>.

For this step, you can assume:
- Both input matrices have valid shapes for multiplication.
- Both matrices are non-empty.
- You do NOT need to handle invalid shapes.

Why this matters for neural networks:
A dense layer computes:

output = input * weights

So before building a neural network, you need to be comfortable multiplying
matrices in code, not just by hand.

Shape rules (for your intuition, even if you assume valid input here):
If A has shape (m x n) and B has shape (n x p),
then A * B is valid and the result has shape (m x p).

In other words:
- columns of A = rows of B
- result rows = rows of A
- result cols = cols of B

How matrix multiplication works:
To compute result[i][j]:
1. Take row i from A
2. Take column j from B
3. Multiply matching entries
4. Add them up

Formula:
result[i][j] = sum of A[i][k] * B[k][j] for all k

How this feels on paper vs in code:
On paper, you usually look at one row and one column at a time.
In code, that becomes 3 loops:
- outer loop over result rows
- middle loop over result cols
- inner loop over k for the dot product

Examples:

Input:
A = [[1.0, 2.0],
     [3.0, 4.0]]

B = [[5.0, 6.0],
     [7.0, 8.0]]

Output:
[[19.0, 22.0],
 [43.0, 50.0]]

Explanation:
result[0][0] = 1*5 + 2*7 = 19
result[0][1] = 1*6 + 2*8 = 22
result[1][0] = 3*5 + 4*7 = 43
result[1][1] = 3*6 + 4*8 = 50

Input:
A = [[1.0, 2.0, 3.0]]

B = [[4.0],
     [5.0],
     [6.0]]

Output:
[[32.0]]

Explanation:
result[0][0] = 1*4 + 2*5 + 3*6 = 32

Input:
A = [[2.0, 0.0],
     [1.0, 3.0]]

B = [[1.0, 4.0],
     [2.0, 5.0]]

Output:
[[2.0, 8.0],
 [7.0, 19.0]]
*/

/* Helpful Resources:

Vec indexing:
https://doc.rust-lang.org/std/vec/struct.Vec.html

Structs and methods:
https://doc.rust-lang.org/book/ch05-01-defining-structs.html

Matrix multiplication shape rule:
The number of columns in the first matrix must equal the number of rows
in the second matrix.
https://en.wikipedia.org/wiki/Matrix_multiplication

Matrix-vector intuition:
A matrix-vector product takes the dot product of the input vector
with each row.
https://mathinsight.org/matrix_vector_multiplication
*/

struct Solution;

impl Solution {
    pub fn matmul(v1: Vec<Vec<f32>>, v2: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let mut result = vec![vec![0.0; 2]; 2];
        let a = v1[0][0];
        let b = v1[0][1];
        let c = v1[1][0];
        let d = v1[1][1];
        let e = v2[0][0];
        let f = v2[0][1];
        let g = v2[1][0];
        let h = v2[1][1];

        result[0][0] = a*e + b*g;
        result[0][1] = a*f + b*h;
        result[1][0] = c*e + d*g;
        result[1][1] = c*f + d*h;

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
    a: Vec<Vec<f32>>,
    b: Vec<Vec<f32>>,
    expected: Vec<Vec<f32>>,
) {
    let actual = Solution::matmul(a.clone(), b.clone());

    let passed = matrices_close(&actual, &expected, 1e-5);

    let green = "\x1b[32m";
    let red = "\x1b[31m";
    let cyan = "\x1b[36m";
    let yellow = "\x1b[33m";
    let bold = "\x1b[1m";
    let reset = "\x1b[0m";

    println!("\n{}{}{}{}", bold, cyan, case_name, reset);
    println!("{}Input A:{} {:?}", yellow, reset, a);
    println!("{}Input B:{} {:?}", yellow, reset, b);
    println!("{}Expected:{} {:?}", yellow, reset, expected);
    println!("{}Actual:{}   {:?}", yellow, reset, actual);

    if passed {
        println!("{}✅ PASS{}", green, reset);
    } else {
        println!("{}❌ FAIL{}", red, reset);
    }
}

fn main() {
    println!("\x1b[1m\x1b[35mNN Runner: Matrix Multiplication\x1b[0m");

    print_result(
        "Test Case 1",
        vec![vec![1.0, 2.0], vec![3.0, 4.0]],
        vec![vec![5.0, 6.0], vec![7.0, 8.0]],
        vec![vec![19.0, 22.0], vec![43.0, 50.0]],
    );

    print_result(
        "Test Case 2",
        vec![vec![1.0, 2.0, 3.0]],
        vec![vec![4.0], vec![5.0], vec![6.0]],
        vec![vec![32.0]],
    );

    print_result(
        "Test Case 3",
        vec![vec![2.0, 0.0], vec![1.0, 3.0]],
        vec![vec![1.0, 4.0], vec![2.0, 5.0]],
        vec![vec![2.0, 8.0], vec![7.0, 19.0]],
    );
}