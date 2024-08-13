use rand::Rng;
use rustacuda::prelude::*;
use rustacuda::memory::DeviceBuffer;
use std::error::Error;
use std::ffi::CString;
use std::fs::File;
use std::io::{Read, Write};


macro_rules! cuda_check {
    ($expr:expr) => {
        match $expr {
            Ok(res) => res,
            Err(e) => {
                eprintln!("CUDA error: {} at {}:{}", e, file!(), line!());
                return Err(Box::new(e));
            }
        }
    };
}

// Define the 256-bit field element type
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Fe([u32; 8]);

// Implement DeviceCopy for Fe
unsafe impl rustacuda::memory::DeviceCopy for Fe {}

// Implement Pod for Fe
unsafe impl bytemuck::Pod for Fe {}
unsafe impl bytemuck::Zeroable for Fe {}

const POLY_DEGREE: usize = 1 << 23;
const SRS_FILE: &str = "srs.bin";

fn generate_and_store_srs() -> Result<(), Box<dyn Error>> {
    let mut rng = rand::thread_rng();
    let srs: Vec<Fe> = (0..=POLY_DEGREE).map(|_| {
        Fe([rng.gen(), rng.gen(), rng.gen(), rng.gen(),
            rng.gen(), rng.gen(), rng.gen(), rng.gen()])
    }).collect();

    let mut file = File::create(SRS_FILE)?;
    file.write_all(bytemuck::cast_slice(&srs))?;

    println!("SRS generated and stored in {}", SRS_FILE);
    Ok(())
}

fn load_srs() -> Result<Vec<Fe>, Box<dyn Error>> {
    let mut file = File::open(SRS_FILE)?;
    let mut buffer = vec![0u8; (POLY_DEGREE + 1) * std::mem::size_of::<Fe>()];
    file.read_exact(&mut buffer)?;
    
    Ok(bytemuck::cast_slice(&buffer).to_vec())
}

fn generate_large_polynomial() -> Vec<Fe> {
    let mut rng = rand::thread_rng();
    (0..=POLY_DEGREE).map(|_| {
        Fe([rng.gen(), rng.gen(), rng.gen(), rng.gen(),
            rng.gen(), rng.gen(), rng.gen(), rng.gen()])
    }).collect()
}

// CUDA kernels
mod kernels {
    use super::Fe;
    use std::ffi::CString;
    use rustacuda::error::CudaError;
    use rustacuda::memory::DeviceBuffer;
    use rustacuda::stream::Stream;
    use rustacuda::launch;
    use rustacuda::prelude::*;
    #[allow(dead_code)]
    extern "C" 
    {
        
        fn evaluate_polynomial_kernel(coeffs: *mut Fe, degree: u32, x: *const Fe, result: *mut Fe, modulus: *const Fe);
        fn calculate_quotient_polynomial_kernel(p: *mut Fe, degree: u32, b: *const Fe, a: *const Fe, q: *mut Fe, modulus: *const Fe);
        fn perform_msm_kernel(points: *mut Fe, scalars: *mut Fe, size: u32, result: *mut Fe, modulus: *const Fe);
    }

    pub unsafe fn evaluate_polynomial(
        module: &Module,
        stream: &Stream,
        coeffs: &mut DeviceBuffer<Fe>,
        degree: u32,
        x: &Fe,
        result: &mut DeviceBuffer<Fe>,
        modulus: &Fe,
    ) -> rustacuda::error::CudaResult<()> {
        let func_name = CString::new("evaluate_polynomial_kernel").map_err(|_| CudaError::UnknownError)?;
        let func = module.get_function(&func_name)?;
        let (grid_size, block_size) = (16, 16);
        
        let mut x_buffer = DeviceBuffer::from_slice(&[*x])?;
        let mut modulus_buffer = DeviceBuffer::from_slice(&[*modulus])?;
        
        launch!(func<<<grid_size, block_size, 0, stream>>>(
            coeffs.as_device_ptr(),
            degree,
            x_buffer.as_device_ptr(),
            result.as_device_ptr(),
            modulus_buffer.as_device_ptr()
        ))?;
        
        Ok(())
    }

    pub unsafe fn calculate_quotient_polynomial(
        module: &Module,
        stream: &Stream,
        p: &mut DeviceBuffer<Fe>,
        degree: u32,
        b: &Fe,
        a: &Fe,
        q: &mut DeviceBuffer<Fe>,
        modulus: &Fe,
    ) -> rustacuda::error::CudaResult<()> {
        let func_name = CString::new("calculate_quotient_polynomial_kernel").map_err(|_| CudaError::UnknownError)?;
        let func = module.get_function(&func_name)?;
        let block_size = 256;
        let grid_size = (degree as u32 + block_size - 1) / block_size;
        
        let mut b_buffer = DeviceBuffer::from_slice(&[*b])?;
        let mut a_buffer = DeviceBuffer::from_slice(&[*a])?;
        let mut modulus_buffer = DeviceBuffer::from_slice(&[*modulus])?;
        
        launch!(func<<<grid_size, block_size, 0, stream>>>(
            p.as_device_ptr(),
            degree,
            b_buffer.as_device_ptr(),
            a_buffer.as_device_ptr(),
            q.as_device_ptr(),
            modulus_buffer.as_device_ptr()
        ))?;
        
        Ok(())
    }

    pub unsafe fn perform_msm(
        module: &Module,
        stream: &Stream,
        points: &mut DeviceBuffer<Fe>,
        scalars: &mut DeviceBuffer<Fe>,
        size: u32,
        result: &mut DeviceBuffer<Fe>,
        modulus: &Fe,
    ) -> rustacuda::error::CudaResult<()> {
        let func_name = CString::new("perform_msm_kernel").map_err(|_| CudaError::UnknownError)?;
        let func = module.get_function(&func_name)?;
        let (grid_size, block_size) = (16, 16);
        
        let mut modulus_buffer = DeviceBuffer::from_slice(&[*modulus])?;
        
        launch!(func<<<grid_size, block_size, 0, stream>>>(
            points.as_device_ptr(),
            scalars.as_device_ptr(),
            size,
            result.as_device_ptr(),
            modulus_buffer.as_device_ptr()
        ))?;
        
        Ok(())
    }
}

struct KZGProof {
    commitment: Vec<Fe>,
    evaluation: Fe,
    proof: Vec<Fe>,
}

fn generate_kzg_proof(polynomial: &[Fe], srs: &[Fe], modulus: &Fe) -> Result<KZGProof, Box<dyn Error>> {
    println!("Initializing CUDA...");
    cuda_check!(rustacuda::init(CudaFlags::empty()));
    
    let device = cuda_check!(Device::get_device(0));
    println!("Using device: {}", device.name()?);

    let _context = cuda_check!(Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device));

    println!("Loading CUDA module...");
    let ptx = CString::new(include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx")))?;
    let module = cuda_check!(Module::load_from_string(&ptx));
    
    println!("Creating CUDA stream...");
    let stream = cuda_check!(Stream::new(StreamFlags::NON_BLOCKING, None));

    println!("Preparing device buffers...");
    let degree = polynomial.len() as u32 - 1;
    let mut d_polynomial = cuda_check!(DeviceBuffer::from_slice(polynomial));
    let mut d_result = cuda_check!(DeviceBuffer::from_slice(&[Fe([0; 8])]));

    println!("Generating random scalar...");
    let mut rng = rand::thread_rng();
    let a = Fe([rng.gen(), rng.gen(), rng.gen(), rng.gen(),
                rng.gen(), rng.gen(), rng.gen(), rng.gen()]);

    println!("Evaluating polynomial...");
    unsafe {
        cuda_check!(kernels::evaluate_polynomial(&module, &stream, &mut d_polynomial, degree, &a, &mut d_result, modulus));
    }
    
    println!("Synchronizing stream...");
    cuda_check!(stream.synchronize());

    println!("Copying evaluation result from device to host...");
    let mut evaluation_host = vec![Fe([0; 8])];
    cuda_check!(d_result.copy_to(&mut evaluation_host));
    let evaluation = evaluation_host[0];

    println!("Calculating quotient polynomial...");
    let mut d_quotient = cuda_check!(DeviceBuffer::from_slice(&vec![Fe([0; 8]); polynomial.len() - 1]));
    unsafe {
        cuda_check!(kernels::calculate_quotient_polynomial(&module, &stream, &mut d_polynomial, degree, &evaluation, &a, &mut d_quotient, modulus));
    }
    cuda_check!(stream.synchronize());

    println!("Performing MSM for commitment...");
    let mut d_srs = cuda_check!(DeviceBuffer::from_slice(srs));
    let mut d_commitment = cuda_check!(DeviceBuffer::from_slice(&[Fe([0; 8]); 2]));
    unsafe {
        cuda_check!(kernels::perform_msm(&module, &stream, &mut d_srs, &mut d_polynomial, polynomial.len() as u32, &mut d_commitment, modulus));
    }
    cuda_check!(stream.synchronize());

    println!("Performing MSM for proof...");
    let mut d_proof = cuda_check!(DeviceBuffer::from_slice(&[Fe([0; 8]); 2]));
    unsafe {
        cuda_check!(kernels::perform_msm(&module, &stream, &mut d_srs, &mut d_quotient, (polynomial.len() - 1) as u32, &mut d_proof, modulus));
    }
    cuda_check!(stream.synchronize());

    println!("Copying results from device to host...");
    let mut commitment = vec![Fe([0; 8]); 2];
    let mut proof = vec![Fe([0; 8]); 2];
    cuda_check!(d_commitment.copy_to(&mut commitment));
    cuda_check!(d_proof.copy_to(&mut proof));

    println!("CUDA operations completed successfully.");

    Ok(KZGProof {
        commitment,
        evaluation,
        proof,
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    // Check if SRS file exists, if not, generate it
    if !std::path::Path::new(SRS_FILE).exists() {
        println!("SRS file not found. Generating new SRS...");
        generate_and_store_srs()?;
    }

    println!("Loading SRS...");
    let srs = load_srs()?;

    println!("Generating large polynomial...");
    let polynomial = generate_large_polynomial();
    
    let modulus = Fe([0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
                      0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x3FFFFFFF]);

    println!("Generating KZG proof for polynomial of degree {}...", POLY_DEGREE);
    let start = std::time::Instant::now();
    let proof = generate_kzg_proof(&polynomial, &srs, &modulus)?;
    let duration = start.elapsed();

    println!("KZG Proof generated successfully");
    println!("Evaluation: {:?}", proof.evaluation);
    println!("Commitment: {:?}", proof.commitment);
    println!("Proof: {:?}", proof.proof);
    println!("Time taken: {:?}", duration);

    Ok(())
}
