import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt

# --- Bagian 1: Logika Inti Otsu (Fungsi Fitness) ---
# Fungsi ini menghitung varians antar-kelas (between-class variance)
# yang akan menjadi nilai fitness untuk setiap kromosom (threshold).
# Semakin tinggi nilainya, semakin bagus (menggunakan maksimasi Otsu).
def otsu_fitness_function(histogram, threshold):
    """Menghitung fitness (varians antar-kelas) untuk nilai threshold tertentu."""
    
    total_pixels = np.sum(histogram)
    
    # Hitung probabilitas kelas 1 (background)
    w1 = np.sum(histogram[:threshold]) / total_pixels
    if w1 == 0:
        return 0 # Hindari pembagian dengan nol

    # Hitung probabilitas kelas 2 (foreground)
    w2 = np.sum(histogram[threshold:]) / total_pixels
    if w2 == 0:
        return 0 # Hindari pembagian dengan nol

    # Hitung rata-rata intensitas kelas 1
    mean1 = np.sum(np.arange(threshold) * histogram[:threshold]) / np.sum(histogram[:threshold])
    
    # Hitung rata-rata intensitas kelas 2
    mean2 = np.sum(np.arange(threshold, 256) * histogram[threshold:]) / np.sum(histogram[threshold:])

    # Hitung varians antar-kelas
    between_class_variance = w1 * w2 * ((mean1 - mean2) ** 2)
    
    return between_class_variance

# --- Bagian 2: Implementasi Genetic Algorithm Manual ---

# Representasi kromosom: nilai threshold 0-255 direpresentasikan sbg 8-bit biner.
def to_binary(n):
    """Mengubah desimal ke 8-bit biner."""
    return format(n, '08b')

def to_decimal(b):
    """Mengubah biner ke desimal."""
    return int(b, 2)

def create_initial_population(pop_size):
    """Membuat populasi awal secara acak."""
    population = []
    for _ in range(pop_size):
        # Buat individu acak (nilai threshold antara 0 dan 255)
        random_decimal = random.randint(0, 255)
        population.append(to_binary(random_decimal))
    return population

def selection(population, fitness_scores):
    """Memilih induk menggunakan Tournament Selection."""
    tournament_size = 3
    # Pilih beberapa individu secara acak untuk turnamen
    tournament_contenders_indices = random.sample(range(len(population)), tournament_size)
    
    # Cari pemenang turnamen (fitness tertinggi)
    winner_index = tournament_contenders_indices[0]
    for i in tournament_contenders_indices:
        if fitness_scores[i] > fitness_scores[winner_index]:
            winner_index = i
            
    return population[winner_index]

def crossover(parent1, parent2, crossover_rate):
    """Melakukan single-point crossover."""
    if random.random() < crossover_rate:
        # Pilih titik potong acak
        point = random.randint(1, 7)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    # Jika tidak crossover, anak sama dengan induk
    return parent1, parent2

def mutation(chromosome, mutation_rate):
    """Melakukan mutasi dengan membalik bit."""
    mutated_chromosome = list(chromosome)
    for i in range(len(mutated_chromosome)):
        if random.random() < mutation_rate:
            # Balik bit (0 menjadi 1, 1 menjadi 0)
            mutated_chromosome[i] = '1' if mutated_chromosome[i] == '0' else '0'
    return "".join(mutated_chromosome)

# --- Bagian 3: Proses Utama ---

def run_ga_otsu(image_path, pop_size=20, generations=100, crossover_rate=0.8, mutation_rate=0.02):
    """Menjalankan seluruh proses GA untuk mencari threshold Otsu."""

    print("--- Memulai Genetic Algorithm untuk Otsu Thresholding ---")

    try:
        img = Image.open(image_path).convert('L')  # 'L' untuk grayscale
        img_np = np.array(img)
        histogram, _ = np.histogram(img_np, bins=256, range=(0, 256))
    except FileNotFoundError:
        print(f"Error: File gambar tidak ditemukan di '{image_path}'")
        return

    population = create_initial_population(pop_size)
    best_overall_threshold = 0
    best_overall_fitness = -1
    no_improve_count = 0
    best_fitnesses = []  # Untuk plotting konvergensi

    for gen in range(generations):
        fitness_scores = [otsu_fitness_function(histogram, to_decimal(chromo)) for chromo in population]
        best_gen_fitness = max(fitness_scores)
        best_gen_idx = fitness_scores.index(best_gen_fitness)
        best_gen_threshold = to_decimal(population[best_gen_idx])

        if best_gen_fitness > best_overall_fitness:
            best_overall_fitness = best_gen_fitness
            best_overall_threshold = best_gen_threshold
            no_improve_count = 0
        else:
            no_improve_count += 1

        best_fitnesses.append(best_overall_fitness)

        # Cek jumlah threshold unik (diversitas)
        unique_thresholds = len(set([to_decimal(ind) for ind in population]))
        print(f"\nGenerasi {gen+1:03d} | Threshold Terbaik: {best_overall_threshold:03d} | Fitness: {best_overall_fitness:.4f} | Threshold Unik: {unique_thresholds}")

        if no_improve_count >= 10:
            print(f"\nEarly stopping: Tidak ada peningkatan dalam 10 generasi. Menghentikan pada generasi {gen+1}.")
            break

        new_population = [population[best_gen_idx]]  # Elitisme

        print("Pasangan Induk yang Dikawinkan:")
        pasangan_log = []

        while len(new_population) < pop_size:
            parent1 = selection(population, fitness_scores)
            parent2 = selection(population, fitness_scores)
            p1_dec = to_decimal(parent1)
            p2_dec = to_decimal(parent2)
            pasangan_log.append((p1_dec, p2_dec))

            child1, child2 = crossover(parent1, parent2, crossover_rate)
            new_population.append(mutation(child1, mutation_rate))
            if len(new_population) < pop_size:
                new_population.append(mutation(child2, mutation_rate))

        for i, (p1, p2) in enumerate(pasangan_log):
            print(f"  Pasangan-{i+1:02d}: {p1} x {p2}")

        population = new_population

    print("\n--- Proses Selesai ---")
    print(f"Threshold optimal yang ditemukan oleh GA: {best_overall_threshold}")

    # Terapkan threshold ke gambar
    binary_img_np = (img_np >= best_overall_threshold) * 255
    binary_img = Image.fromarray(binary_img_np.astype(np.uint8))

    # Tampilkan hasilnya
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title('Citra Asli')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Histogram')
    plt.hist(img_np.ravel(), bins=256, range=(0, 256), color='blue')
    plt.axvline(best_overall_threshold, color='r', linestyle='dashed', linewidth=2, label=f'Threshold = {best_overall_threshold}')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.title('Hasil Segmentasi (GA-Otsu)')
    plt.imshow(binary_img, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Plot konvergensi fitness
    plt.figure(figsize=(8, 4))
    plt.plot(best_fitnesses, marker='o')
    plt.title("Plot Konvergensi Fitness Terbaik per Generasi")
    plt.xlabel("Generasi")
    plt.ylabel("Fitness Terbaik")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# --- Jalankan Program ---
if __name__ == '__main__':
    # Ganti 'sample_image.jpg' dengan path gambar Anda
    IMAGE_FILE_PATH = "e2627990ea6b2c54c9082ee6cfcffd56.jpg" 
    run_ga_otsu(IMAGE_FILE_PATH)