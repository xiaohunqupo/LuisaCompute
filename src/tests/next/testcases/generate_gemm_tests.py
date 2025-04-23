import numpy as np
import json
import os
import argparse 

def generate_gemm_test_cases(output_dir="gemm_test_cases"):
    """Generate various GEMM test cases and save them as JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    test_cases = []
    
    # Test case types
    sizes = [(2, 2, 2), (3, 4, 2), (16, 16, 16), (32, 64, 128), (100, 50, 75)]
    dtypes = [np.float32, np.float64]
    
    case_id = 0
    
    # 1. Standard matrix multiplications with various sizes
    for m, n, k in sizes:
        for dtype in dtypes:
            # Random matrices
            A = np.random.random((m, k)).astype(dtype)
            B = np.random.random((k, n)).astype(dtype)
            C = np.matmul(A, B)
            
            case_name= f"case_{case_id}"
            case_path = os.path.join(output_dir, case_name)
            os.makedirs(case_path, exist_ok=True)
            
            # Save as JSON instead of NPY
            with open(os.path.join(case_path, "A.json"), "w") as f:
                json.dump(A.tolist(), f)
            
            with open(os.path.join(case_path, "B.json"), "w") as f:
                json.dump(B.tolist(), f)
            
            with open(os.path.join(case_path, "C_expected.json"), "w") as f:
                json.dump(C.tolist(), f)
            
            test_cases.append({
                "id": case_id,
                "type": "random",
                "m": m,
                "n": n,
                "k": k,
                "dtype": str(dtype),
                "name": case_name
            })
            case_id += 1
    
    # 2. Identity matrix tests
    for m, n, k in [(10, 10, 10), (50, 50, 50)]:
        for dtype in dtypes:
            # A * Identity = A
            A = np.random.random((m, k)).astype(dtype)
            B = np.eye(k, n, dtype=dtype)
            C = np.matmul(A, B)
            case_name = f"case_{case_id}"
            case_path = os.path.join(output_dir, case_name)
            os.makedirs(case_path, exist_ok=True)
            
            with open(os.path.join(case_path, "A.json"), "w") as f:
                json.dump(A.tolist(), f)
            
            with open(os.path.join(case_path, "B.json"), "w") as f:
                json.dump(B.tolist(), f)
            
            with open(os.path.join(case_path, "C_expected.json"), "w") as f:
                json.dump(C.tolist(), f)
            
            test_cases.append({
                "id": case_id,
                "type": "identity_right",
                "m": m,
                "n": n,
                "k": k,
                "dtype": str(dtype),
                "name": case_name
            })
            case_id += 1
            
            # Identity * B = B
            A = np.eye(m, k, dtype=dtype)
            B = np.random.random((k, n)).astype(dtype)
            C = np.matmul(A, B)
            
            case_path = os.path.join(output_dir, f"case_{case_id}")
            os.makedirs(case_path, exist_ok=True)
            
            with open(os.path.join(case_path, "A.json"), "w") as f:
                json.dump(A.tolist(), f)
            
            with open(os.path.join(case_path, "B.json"), "w") as f:
                json.dump(B.tolist(), f)
            
            with open(os.path.join(case_path, "C_expected.json"), "w") as f:
                json.dump(C.tolist(), f)
            
            test_cases.append({
                "id": case_id,
                "type": "identity_left",
                "m": m,
                "n": n,
                "k": k,
                "dtype": str(dtype),
                "name": case_name
            })
            case_id += 1
    
    # 3. Specific pattern matrices
    for m, n, k in [(8, 8, 8), (32, 32, 32)]:
        for dtype in dtypes:
            # Ones matrices
            A = np.ones((m, k), dtype=dtype)
            B = np.ones((k, n), dtype=dtype)
            C = np.matmul(A, B)
            
            case_name = f"case_{case_id}"
            case_path = os.path.join(output_dir, case_name)
            os.makedirs(case_path, exist_ok=True)
            
            with open(os.path.join(case_path, "A.json"), "w") as f:
                json.dump(A.tolist(), f)
            
            with open(os.path.join(case_path, "B.json"), "w") as f:
                json.dump(B.tolist(), f)
            
            with open(os.path.join(case_path, "C_expected.json"), "w") as f:
                json.dump(C.tolist(), f)
            
            test_cases.append({
                "id": case_id,
                "type": "ones",
                "m": m,
                "n": n,
                "k": k,
                "dtype": str(dtype),
                "name": case_name
            })
            case_id += 1
            
            # Diagonal matrices
            A = np.diag(np.random.random(min(m, k)).astype(dtype))
            if m > k:
                A = np.vstack([A, np.zeros((m-k, k), dtype=dtype)])
            elif k > m:
                A = np.hstack([A, np.zeros((m, k-m), dtype=dtype)])
                
            B = np.diag(np.random.random(min(k, n)).astype(dtype))
            if k > n:
                B = np.hstack([B, np.zeros((k, n-n), dtype=dtype)])
            elif n > k:
                B = np.vstack([B, np.zeros((n-k, n), dtype=dtype)])
                
            C = np.matmul(A, B)
            
            case_path = os.path.join(output_dir, f"case_{case_id}")
            os.makedirs(case_path, exist_ok=True)
            
            with open(os.path.join(case_path, "A.json"), "w") as f:
                json.dump(A.tolist(), f)
            
            with open(os.path.join(case_path, "B.json"), "w") as f:
                json.dump(B.tolist(), f)
            
            with open(os.path.join(case_path, "C_expected.json"), "w") as f:
                json.dump(C.tolist(), f)
            
            test_cases.append({
                "id": case_id,
                "type": "diagonal",
                "m": m,
                "n": n,
                "k": k,
                "dtype": str(dtype),
                "name": case_name
            })
            case_id += 1
    
    # 4. Near-zero values test
    for dtype in dtypes:
        m, n, k = 16, 16, 16
        A = np.random.random((m, k)).astype(dtype) * 1e-6
        B = np.random.random((k, n)).astype(dtype) * 1e-6
        C = np.matmul(A, B)
        
        case_name = f"case_{case_id}"
        case_path = os.path.join(output_dir, case_name)
        os.makedirs(case_path, exist_ok=True)
        
        with open(os.path.join(case_path, "A.json"), "w") as f:
            json.dump(A.tolist(), f)
        
        with open(os.path.join(case_path, "B.json"), "w") as f:
            json.dump(B.tolist(), f)
        
        with open(os.path.join(case_path, "C_expected.json"), "w") as f:
            json.dump(C.tolist(), f)
        
        test_cases.append({
            "id": case_id,
            "type": "small_values",
            "m": m,
            "n": n,
            "k": k,
            "dtype": str(dtype),
            "name": case_name
        })
        case_id += 1
    
    # Save test case metadata
    with open(os.path.join(output_dir, "test_cases.json"), "w") as f:
        json.dump(test_cases, f, indent=2)
    
    print(f"Generated {len(test_cases)} GEMM test cases in '{output_dir}'")
    return test_cases

def load_test_cases(input_dir="gemm_test_cases"):
    """Load the generated test cases from JSON files."""
    with open(os.path.join(input_dir, "test_cases.json"), "r") as f:
        test_cases = json.load(f)
    
    loaded_cases = []
    for case in test_cases:
        # Load JSON files and convert back to NumPy arrays
        with open(os.path.join(case["path"], "A.json"), "r") as f:
            A_list = json.load(f)
            A = np.array(A_list)
        
        with open(os.path.join(case["path"], "B.json"), "r") as f:
            B_list = json.load(f)
            B = np.array(B_list)
        
        with open(os.path.join(case["path"], "C_expected.json"), "r") as f:
            C_list = json.load(f)
            C_expected = np.array(C_list)
        
        # Convert to the correct dtype
        if "float32" in case["dtype"]:
            A = A.astype(np.float32)
            B = B.astype(np.float32)
            C_expected = C_expected.astype(np.float32)
        elif "float64" in case["dtype"]:
            A = A.astype(np.float64)
            B = B.astype(np.float64)
            C_expected = C_expected.astype(np.float64)
        
        loaded_cases.append({
            "id": case["id"],
            "type": case["type"],
            "m": case["m"],
            "n": case["n"],
            "k": case["k"],
            "dtype": case["dtype"],
            "A": A,
            "B": B,
            "C_expected": C_expected
        })
    
    return loaded_cases

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GEMM test cases")
    parser.add_argument("--output_dir", type=str, default="gemm_test_cases", help="Output directory for test cases")
    args = parser.parse_args()
    generate_gemm_test_cases(output_dir=args.output_dir)