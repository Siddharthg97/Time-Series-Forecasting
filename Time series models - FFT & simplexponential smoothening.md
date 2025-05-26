**Baseline models**
i) ETS - Single exponential smoothening




**Fourier Transform**
The Fourier Transform is a powerful tool in time series analysis that helps you understand the frequency components of a signal or dataset. Here's a quick breakdown of what it is and how it's used:


🧠 What Is the Fourier Transform?
The Fourier Transform decomposes a time series (or signal) into a sum of sine and cosine waves at different frequencies. In other words:

It tells you which frequencies are present in your time series, and how strong they are.

🔁 Why Use It in Time Series?
Time series often have hidden cycles or periodic behavior. Fourier analysis helps you:

Detect seasonal patterns (daily, weekly, annual cycles).

Filter out noise.

Transform data for spectral analysis.

Identify dominant frequencies or repeating trends.


🔎 Key Terms
FFT (Fast Fourier Transform): Efficient algorithm for computing the Fourier Transform.

Frequency domain: The transformed space showing frequency instead of time.

Power Spectrum: Shows the intensity (power) of each frequency component.

⚠️ Limitations
Assumes stationarity (i.e., signal doesn't change over time).

Doesn’t capture transient or non-stationary features well.

For that, consider Short-Time Fourier Transform (STFT) or Wavelet Transforms.

**Introduction of fourier terms in Regression models**

def compute_fourier_terms(frequencies, magnitudes, phases, num_points):
    """
    Generates Fourier terms (sine and cosine) for each frequency with a default of 10 time points.

    :param frequencies: List of frequencies.
    :param magnitudes: List of corresponding magnitudes.
    :param phases: List of corresponding phases.
    :param num_points: Number of time points (length of the time series), default is 10.
    :return: List of Fourier terms (each term is a list over time).
    """
    time = np.arange(num_points)  # Time steps

    fourier_terms = []

    for freq, mag, phase in zip(frequencies, magnitudes, phases):
        if freq == 0 or mag ==0:
            continue
        # Compute cosine for each time step
        cosine_term = [float(mag * np.cos(2 * np.pi * freq * t + phase)) for t in time]
        sine_term = [float(mag * np.sin(2 * np.pi * freq * t + phase)) for t in time]

        # Append both terms
        fourier_terms.append(cosine_term)
        fourier_terms.append(sine_term)

    return fourier_terms


# Register UDF correctly
fft_udf = F.udf(lambda values: compute_fft(values), ArrayType(ArrayType(DoubleType())))

# UDF to apply the Fourier term computation
fourier_terms_udf = F.udf(compute_fourier_terms, ArrayType(ArrayType(DoubleType())))

def add_fourier_terms(df: DataFrame, threshold=3.0):
    """
    Computes the FFT for each 'Key' group in the PySpark DataFrame.

    :param df: PySpark DataFrame with 'BusinessDate', 'Actual', and 'Key' columns.
    :param threshold: Z-score threshold for filtering outliers.
    :return: DataFrame with filtered frequencies, magnitudes, Fourier terms and phases for each Key.
    """
    # Group by 'Key' and collect 'Actual' values
    df_grouped = df.groupBy("Key").agg(
    F.collect_list(F.col("Actual")).alias("values"),   # Collects all 'Actual' values into a list per 'Key'
    F.count(F.col("Key")).alias("Key_counts"))          # Counts the number of occurrences of each 'Key'

    # Apply the FFT UDF correctly
    df_fft = df_grouped.withColumn("fft_results", fft_udf(F.col("values")))

    # Extract frequencies, magnitudes, and phases correctly
    df_fft = df_fft.withColumn("frequencies", F.col("fft_results")[0]) \
                   .withColumn("magnitudes", F.col("fft_results")[1]) \
                   .withColumn("phases", F.col("fft_results")[2]) \
                   .drop("fft_results", "values")
    # Apply the UDF to compute Fourier terms
    df_with_fourier = df_fft.withColumn(
        "fourier_terms",
        fourier_terms_udf(F.col("frequencies"), F.col("magnitudes"), F.col("phases"), F.col("Key_counts"))
    )

    # Determine the maximum length of the lists in the 'fourier_terms' column
    max_len = df_with_fourier.select(F.max(F.size('fourier_terms'))).collect()[0][0]

    # Create a new column for each item in the list
    for i in range(max_len):
        df_with_fourier = df_with_fourier.withColumn(f"fourier_cos_{i+1}", F.col('fourier_terms')[i])
    
    return df_with_fourier.drop("frequencies", "magnitudes", "phases", "fourier_terms", "Key_counts")

 **Why Use Fourier Terms in Regression?**
To model seasonality (daily, weekly, yearly cycles) explicitly.

More flexible and parsimonious than dummy variables for cycles.

Especially useful when using linear models that don’t handle time-dependence naturally.

def create_fourier_terms(t, period, K):
    terms = {}
    for k in range(1, K + 1):
        terms[f'sin_{k}'] = np.sin(2 * np.pi * k * t / period)
        terms[f'cos_{k}'] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(terms)

# Add Fourier terms to dataframe
K = 2  # number of harmonics
period = 365
fourier_df = create_fourier_terms(df['day'], period, K)
df = pd.concat([df, fourier_df], axis=1)

