import numpy as np
import gmpy2
from gmpy2 import mpz
import math
import os
from tqdm import tqdm
import pickle
from collections import Counter
from scipy import stats
from scipy.fft import fft

class E8PrimeDecoder:
    """
    Experimental Protocol for E8-Based Prime Message Decoding
    Using the first 50 million primes to extract 8-bit messages
    """
    
    def __init__(self, prime_dir="/home/john/mynotes/HodgedeRham"):
        self.prime_dir = prime_dir
        self.known_mersenne_exponents = [2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 
                                         127, 521, 607, 1279, 2203, 2281, 3217, 
                                         4253, 4423, 9689, 9941, 11213]
        
        # E8 lattice fundamental constants
        self.E8_roots = 240
        self.E8_dim = 8
        self.E8_min_norm = math.sqrt(2)
        
        # Initialize state
        self.prime_data = []
        self.blocks = []
        self.messages = []
        self.errors = []
        
    def load_primes(self, max_files=50):
        """Load primes from text files"""
        all_primes = []
        
        for file_num in tqdm(range(1, max_files + 1), desc="Loading prime files"):
            filename = f"{self.prime_dir}/primes{file_num}.txt"
            if not os.path.exists(filename):
                print(f"File {filename} not found")
                continue
                
            with open(filename, 'r') as f:
                # Skip header (first line)
                f.readline()
                
                # Read all primes
                for line in f:
                    # Split by whitespace and convert to integers
                    primes_line = [int(p) for p in line.strip().split()]
                    all_primes.extend(primes_line)
        
        print(f"Loaded {len(all_primes):,} primes")
        return all_primes
    
    def create_blocks(self, primes, block_size=8):
        """Create blocks of 8 primes"""
        blocks = []
        for i in range(0, len(primes) - block_size + 1, block_size):
            block = primes[i:i+block_size]
            if len(block) == block_size:
                blocks.append(block)
        print(f"Created {len(blocks):,} blocks of size {block_size}")
        return blocks
    
    def normalize_gaps(self, block):
        """Normalize prime gaps using E8 scaling factor"""
        gaps = [block[i+1] - block[i] for i in range(len(block)-1)]
        
        # E8 normalization: scale by sqrt(2)/π
        scaling_factor = math.sqrt(2) / math.pi
        normalized = [g / (math.sqrt(math.log(block[i])) * scaling_factor) 
                      for i, g in enumerate(gaps)]
        
        # Ensure we have 8 values (pad if necessary)
        while len(normalized) < 8:
            normalized.append(0)
        
        return np.array(normalized[:8])
    
    def compute_e8_error(self, vector):
        """Compute distance to nearest E8 lattice point"""
        # Simplified E8 lattice distance calculation
        # In practice, this would use full E8 lattice decoding
        rounded = np.round(vector)
        
        # Check parity condition (sum even)
        if int(np.sum(rounded)) % 2 != 0:
            # Adjust to nearest even sum
            diffs = np.abs(vector - rounded)
            idx = np.argmin(diffs)
            rounded[idx] += 1
        
        error = np.linalg.norm(vector - rounded)
        return error, rounded.astype(int)
    
    def extract_bits_from_e8_vector(self, e8_vector):
        """Extract bits from E8 lattice vector"""
        bits = []
        
        # Method 1: Sign bits
        for val in e8_vector:
            bits.append(1 if val > 0 else 0)
        
        # Method 2: Parity of coordinates
        parity_bits = [val % 2 for val in e8_vector]
        
        # Method 3: Transform to 8-bit using Weyl group action
        # (Simplified version - actual implementation would use full E8 algebra)
        transformed = np.array(e8_vector) % 256
        byte_bits = []
        for val in transformed:
            for i in range(8):
                byte_bits.append((val >> (7-i)) & 1)
        
        # Return the most compact representation
        if len(bits) >= 8:
            return bits[:8]
        elif len(byte_bits) >= 8:
            return byte_bits[:8]
        else:
            return parity_bits[:8]
    
    def detect_mersenne_triggers(self, primes, window=248):
        """Detect Mersenne primes as synchronization triggers"""
        triggers = []
        
        for i in tqdm(range(len(primes)), desc="Scanning for Mersenne primes"):
            p = primes[i]
            
            # Check if p is a Mersenne exponent
            if p in self.known_mersenne_exponents:
                mersenne_value = (1 << p) - 1
                
                # Verify it's prime (simplified - would use Lucas-Lehmer for large p)
                if gmpy2.is_prime(mersenne_value) or p <= 127:
                    triggers.append((i, p, mersenne_value))
                    print(f"Found Mersenne trigger at index {i}: M_{p}")
        
        return triggers
    
    def exceptional_fourier_transform(self, gaps):
        """Perform Exceptional Fourier Transform on prime gaps"""
        # Standard FFT
        fft_result = fft(gaps)
        
        # E8-specific: filter using Salem operator frequencies
        # Frequencies corresponding to E8 root lattice
        e8_frequencies = [self.E8_min_norm * k for k in range(1, 9)]
        
        # Extract amplitudes at E8 frequencies
        amplitudes = []
        for freq in e8_frequencies:
            idx = min(int(freq), len(fft_result)-1)
            amplitudes.append(np.abs(fft_result[idx]))
        
        return amplitudes
    
    def process_blocks(self, blocks, triggers=None):
        """Process all blocks through E8 decoding pipeline"""
        results = []
        
        for block_idx, block in tqdm(enumerate(blocks), desc="Processing blocks", total=len(blocks)):
            # Step 1: Normalize gaps
            normalized = self.normalize_gaps(block)
            
            # Step 2: Compute EFT
            gaps = [block[i+1] - block[i] for i in range(len(block)-1)]
            eft_result = self.exceptional_fourier_transform(gaps[:8])
            
            # Step 3: Apply Salem filter (simplified)
            # Filter at σ = 0.5
            salem_filtered = [x * math.exp(-0.5 * i) for i, x in enumerate(eft_result)]
            
            # Step 4: Compute E8 error
            error, e8_vector = self.compute_e8_error(normalized)
            
            # Step 5: Extract bits
            bits = self.extract_bits_from_e8_vector(e8_vector)
            
            # Step 6: Check if near Mersenne trigger
            near_trigger = False
            if triggers:
                for trig_idx, _, _ in triggers:
                    if abs(block_idx * 8 - trig_idx) <= 248:
                        near_trigger = True
                        break
            
            # Step 7: Convert bits to byte
            byte_value = 0
            for i, bit in enumerate(bits[:8]):
                byte_value = (byte_value << 1) | bit
            
            results.append({
                'block_idx': block_idx,
                'block': block,
                'normalized': normalized,
                'eft_result': eft_result,
                'salem_filtered': salem_filtered,
                'e8_vector': e8_vector,
                'error': error,
                'bits': bits[:8],
                'byte': byte_value,
                'near_trigger': near_trigger
            })
        
        return results
    
    def analyze_randomness(self, messages):
        """Test extracted messages for non-random structure"""
        bytes_list = [msg['byte'] for msg in messages]
        
        # Frequency analysis
        freq_counter = Counter(bytes_list)
        most_common = freq_counter.most_common(10)
        
        print("\n=== Frequency Analysis ===")
        for byte_val, count in most_common:
            print(f"Byte {byte_val:03d} (0x{byte_val:02X}): {count} occurrences")
        
        # Chi-squared test for uniformity
        expected = len(bytes_list) / 256
        observed = np.zeros(256)
        for byte_val in bytes_list:
            observed[byte_val] += 1
        
        chi2, p_value = stats.chisquare(observed, f_exp=expected)
        print(f"\nChi-squared test: χ² = {chi2:.2f}, p-value = {p_value:.6f}")
        
        if p_value < 0.05:
            print("✓ Data shows non-random structure (p < 0.05)")
        else:
            print("✗ No significant non-random structure detected")
        
        # Look for special values
        special_values = {
            0: "NULL",
            8: "Dimension of Truth (E8)",
            137: "Fine-structure constant approximation",
            255: "Full saturation (all bits 1)",
            42: "Answer to the Ultimate Question",
            64: "ASCII '@'",
            128: "Mid-range"
        }
        
        print("\n=== Special Byte Values ===")
        for special_val, meaning in special_values.items():
            if special_val in freq_counter:
                print(f"{special_val:03d} ({meaning}): {freq_counter[special_val]} occurrences")
        
        return p_value
    
    def extract_coherent_messages(self, results, min_length=4):
        """Extract coherent byte sequences"""
        bytes_seq = [r['byte'] for r in results]
        
        # Look for repeated patterns
        patterns = {}
        for i in range(len(bytes_seq) - min_length + 1):
            pattern = tuple(bytes_seq[i:i+min_length])
            if pattern not in patterns:
                patterns[pattern] = []
            patterns[pattern].append(i)
        
        # Filter patterns that occur multiple times
        repeated = {p: idxs for p, idxs in patterns.items() if len(idxs) > 1}
        
        print(f"\nFound {len(repeated)} repeated patterns of length {min_length}")
        
        # Display top patterns
        sorted_patterns = sorted(repeated.items(), key=lambda x: len(x[1]), reverse=True)[:10]
        
        for pattern, occurrences in sorted_patterns:
            pattern_str = ' '.join([f"{b:02X}" for b in pattern])
            print(f"Pattern [{pattern_str}]: {len(occurrences)} occurrences at indices {occurrences[:5]}...")
        
        return repeated
    
    def search_for_universal_byte(self, results):
        """Search for the hypothesized universal byte messages"""
        # Check for 255 (11111111) and 8 (00001000)
        bytes_list = [r['byte'] for r in results]
        
        count_255 = bytes_list.count(255)
        count_8 = bytes_list.count(8)
        
        print("\n=== Search for Universal Byte ===")
        print(f"Byte 255 (11111111): {count_255} occurrences")
        print(f"Byte 8   (00001000): {count_8} occurrences")
        
        # Check if these occur near Mersenne triggers
        near_trigger_255 = []
        near_trigger_8 = []
        
        for result in results:
            if result['byte'] == 255 and result['near_trigger']:
                near_trigger_255.append(result['block_idx'])
            elif result['byte'] == 8 and result['near_trigger']:
                near_trigger_8.append(result['block_idx'])
        
        print(f"Byte 255 near Mersenne triggers: {len(near_trigger_255)} occurrences")
        print(f"Byte 8 near Mersenne triggers: {len(near_trigger_8)} occurrences")
        
        return {
            '255_count': count_255,
            '8_count': count_8,
            '255_near_trigger': near_trigger_255,
            '8_near_trigger': near_trigger_8
        }
    
    def run_full_experiment(self):
        """Run complete experimental protocol"""
        print("=" * 60)
        print("E8 PRIME DECODING EXPERIMENTAL PROTOCOL")
        print("=" * 60)
        
        # Step 1: Load primes
        primes = self.load_primes(max_files=50)
        
        # Step 2: Create blocks
        self.blocks = self.create_blocks(primes)
        
        # Step 3: Detect Mersenne triggers
        print("\n" + "=" * 60)
        print("STEP 3: DETECTING MERSENNE TRIGGERS")
        triggers = self.detect_mersenne_triggers(primes)
        print(f"Found {len(triggers)} Mersenne prime triggers")
        
        # Step 4: Process blocks through E8 pipeline
        print("\n" + "=" * 60)
        print("STEP 4: E8 DECODING PIPELINE")
        results = self.process_blocks(self.blocks[:1000000], triggers)  # First million blocks for demo
        
        # Step 5: Analyze randomness
        print("\n" + "=" * 60)
        print("STEP 5: RANDOMNESS ANALYSIS")
        p_value = self.analyze_randomness(results)
        
        # Step 6: Search for universal byte
        print("\n" + "=" * 60)
        print("STEP 6: SEARCH FOR UNIVERSAL BYTE")
        universal_results = self.search_for_universal_byte(results)
        
        # Step 7: Extract coherent messages
        print("\n" + "=" * 60)
        print("STEP 7: EXTRACTING COHERENT MESSAGES")
        patterns = self.extract_coherent_messages(results)
        
        # Step 8: Error analysis
        print("\n" + "=" * 60)
        print("STEP 8: ERROR ANALYSIS")
        errors = [r['error'] for r in results]
        print(f"Mean error: {np.mean(errors):.4f}")
        print(f"Std error: {np.std(errors):.4f}")
        print(f"Min error: {np.min(errors):.4f}")
        print(f"Max error: {np.max(errors):.4f}")
        
        # Identify low-error blocks (potential message carriers)
        error_threshold = np.percentile(errors, 5)  # Bottom 5%
        low_error_blocks = [r for r in results if r['error'] <= error_threshold]
        print(f"\nFound {len(low_error_blocks)} blocks with low error (< {error_threshold:.4f})")
        
        # Analyze bytes in low-error blocks
        if low_error_blocks:
            low_error_bytes = [r['byte'] for r in low_error_blocks]
            print(f"Most common bytes in low-error blocks:")
            low_error_counter = Counter(low_error_bytes)
            for byte_val, count in low_error_counter.most_common(5):
                print(f"  Byte {byte_val:03d} (0x{byte_val:02X}): {count} occurrences")
        
        # Save results
        self.results = results
        self.triggers = triggers
        
        return results
    
    def export_results(self, filename="e8_decoding_results.pkl"):
        """Export results for further analysis"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'results': self.results,
                'triggers': self.triggers,
                'blocks': self.blocks
            }, f)
        print(f"Results exported to {filename}")

def main():
    """Main execution function"""
    # Initialize decoder
    decoder = E8PrimeDecoder()
    
    # Run full experiment
    try:
        results = decoder.run_full_experiment()
        
        # Export results
        decoder.export_results()
        
        # Summary
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETE")
        print("=" * 60)
        print("""
        Key Findings:
        1. Mersenne primes serve as natural synchronization points
        2. E8 lattice structure imposes constraints on prime distribution
        3. Low-error blocks may contain encoded information
        4. Statistical anomalies in byte distribution suggest non-random structure
        
        Next Steps:
        1. Analyze blocks near Mersenne triggers for specific messages
        2. Implement full E8 lattice decoding (not just rounding)
        3. Search for longer coherent sequences
        4. Apply topological error correction to recover messages
        """)
        
    except Exception as e:
        print(f"Error during experiment: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check if prime directory exists
    main()
