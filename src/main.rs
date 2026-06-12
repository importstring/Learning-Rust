/*
LeetCode 73. Set Matrix Zeroes
https://leetcode.com/problems/set-matrix-zeroes/


Given an m x n integer matrix matrix, if an element is 0, set its entire row and column to 0's.

You must do it in place.


Examples:

Input:  matrix = [[1,1,1],[1,0,1],[1,1,1]]
Output: [[1,0,1],[0,0,0],[1,0,1]]

Input:  matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
Output: [[0,0,0,0],[0,4,5,0],[0,3,1,0]]
*/


/* Helpful Resources:

Vec / 2D Vec:
https://doc.rust-lang.org/std/vec/struct.Vec.html

Indexing and lengths:
matrix.len()          // number of rows
matrix[0].len()       // number of columns (if at least 1 row)

Loops over indices:
for i in 0..matrix.len() { ... }
for j in 0..matrix[0].len() { ... }

Creating helper arrays:
let mut rows = vec![false; matrix.len()];
let mut cols = vec![false; matrix[0].len()];

Mutable access:
matrix[i][j] = 0;
*/


struct Solution;

impl Solution {
    pub fn set_zeroes(matrix: &mut Vec<Vec<i32>>) {
        let mut zereod_cols: Vec<usize> = vec![];

        for (idx, row) in matrix.iter().enumerate() {
            if row.contains(&0) {
                matrix[idx] = row.fill(0);
                zereod_cols.push(
                    row.iter().position(|&z| z == 0)
                );
            }
            else {
                for col in zereod_cols.iter() {
                    matrix[idx][col] = 0
                }
            }
        
        matrix

            

        }
    }
}



// Tests


fn print_result(case_name: &str, mut matrix: Vec<Vec<i32>>, expected: Vec<Vec<i32>>) {
    Solution::set_zeroes(&mut matrix);
    let passed = matrix == expected;

    let green = "\x1b[32m";
    let red = "\x1b[31m";
    let cyan = "\x1b[36m";
    let yellow = "\x1b[33m";
    let bold = "\x1b[1m";
    let reset = "\x1b[0m";

    println!("\n{}{}{}{}", bold, cyan, case_name, reset);
    println!("{}Input:{} {:?}", yellow, reset, expected); // you can tweak to print original if you want
    println!("{}Expected:{} {:?}", yellow, reset, expected);
    println!("{}Actual:{}   {:?}", yellow, reset, matrix);

    if passed {
        println!("{}✅ PASS{}", green, reset);
    } else {
        println!("{}❌ FAIL{}", red, reset);
    }
}


fn main() {
    println!("\x1b[1m\x1b[35mLeetCode Runner: Set Matrix Zeroes\x1b[0m");

    print_result(
        "Test Case 1",
        vec![vec![1, 1, 1], vec![1, 0, 1], vec![1, 1, 1]],
        vec![vec![1, 0, 1], vec![0, 0, 0], vec![1, 0, 1]],
    );

    print_result(
        "Test Case 2",
        vec![vec![0, 1, 2, 0], vec![3, 4, 5, 2], vec![1, 3, 1, 5]],
        vec![vec![0, 0, 0, 0], vec![0, 4, 5, 0], vec![0, 3, 1, 0]],
    );

    print_result(
        "Test Case 3",
        vec![vec![1]],
        vec![vec![1]],
    );

    print_result(
        "Test Case 4",
        vec![vec![0, 1]],
        vec![vec![0, 0]],
    );
}