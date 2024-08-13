use std::process::Command;
use std::env;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let cuda_dir = env::var("CUDA_PATH").unwrap_or_else(|_| "/usr/local/cuda".to_string());

    // Compile CUDA kernel
    let status = Command::new("nvcc")
        .args(&[
            "src/kernels.cu",
            "-o",
            &format!("{}/kernels.ptx", out_dir),
            "--ptx",
            "-arch=sm_86", // Adjust this based on your GPU architecture
            &format!("-I{}/include", cuda_dir),
            &format!("-L{}/lib64", cuda_dir),
        ])
        .status()
        .expect("Failed to execute nvcc command");

    if !status.success() {
        panic!("Failed to compile CUDA kernels");
    }

    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-search=native={}/lib64", cuda_dir);
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");

    println!("cargo:rerun-if-changed=src/kernels.cu");
    println!("cargo:rerun-if-changed=build.rs");
}