import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
import re
import warnings

warnings.filterwarnings('ignore')
from gensim.models import FastText
from gensim.utils import simple_preprocess

# Загрузка данных
df = pd.read_csv('customer_data_test.csv')
df = df.iloc[:, :]
df = df.dropna()

print(df.head())
print(df.dtypes)

model = FastText.load("fasttext_lee_background")
VECTOR_SIZE = model.vector_size
MAX_TOKENS_PER_ENTRY = 32

def get_vectorised_entries(entries):
    """
    Векторизация текста
    """
    features = []
    for entry in entries:
        tokens = simple_preprocess(str(entry), min_len=1)
        if tokens:
            vectors = []
            for token in tokens:
                if token in model.wv:
                    vectors.append(model.wv[token])
                else:
                    vectors.append(np.random.randn(VECTOR_SIZE))
            if vectors:
                features.append(np.mean(vectors, axis=0))
            else:
                features.append(np.zeros(VECTOR_SIZE))
        else:
            features.append(np.zeros(VECTOR_SIZE))
    return np.array(features)


def vectorise_entries(entries):
    features = []
    for entry in entries:
        tokens = simple_preprocess(str(entry))
        vectors = []
        for token in tokens:
            if token in model.wv:
                vectors.append(model.wv[token])
            else:
                vectors.append(np.random.randn(VECTOR_SIZE))

        length = min(len(vectors), MAX_TOKENS_PER_ENTRY)
        trimmed_vectors = vectors[:length]

        if length < MAX_TOKENS_PER_ENTRY:
            padding_vectors = [np.zeros(VECTOR_SIZE) for _ in range(MAX_TOKENS_PER_ENTRY - length)]
            trimmed_vectors += padding_vectors

        # Объединяем все векторы в один большой вектор
        concatenated_vector = np.concatenate(trimmed_vectors)
        features.append(concatenated_vector)

    return np.array(features)


def detect_special_format(series):
    """
    Определяет специальные форматы данных в колонке
    """
    series_clean = series.dropna()

    if len(series_clean) == 0:
        return 'general_text'

    email_count = 0
    for val in series_clean.head(100):
        str_val = str(val).strip()
        if '@' in str_val and '.' in str_val.split('@')[-1]:
            email_count += 1

    if email_count / min(100, len(series_clean)) > 0.7:
        return 'email'

    phone_count = 0
    for val in series_clean.head(100):
        str_val = str(val).strip()
        digits_only = re.sub(r'\D', '', str_val)
        if 7 <= len(digits_only) <= 15:
            phone_count += 1

    if phone_count / min(100, len(series_clean)) > 0.7:
        return 'phone'

    return 'general_text'


def detect_boolean_format(series):
    """
    Определяет формат boolean значений
    """
    series_clean = series.dropna()

    if len(series_clean) == 0:
        return None

    formats = {
        'true_false': 0,
        'yes_no': 0,
        'one_zero': 0,
        'yn': 0,
        'tf': 0,
        'bool': 0
    }

    for val in series_clean.head(100):
        str_val = str(val).strip().lower()

        if val is True or val is False:
            formats['bool'] += 1
        elif str_val in ['true', 'false']:
            formats['true_false'] += 1
        elif str_val in ['yes', 'no']:
            formats['yes_no'] += 1
        elif str_val in ['1', '0']:
            formats['one_zero'] += 1
        elif str_val in ['y', 'n']:
            formats['yn'] += 1
        elif str_val in ['t', 'f']:
            formats['tf'] += 1

    max_format = max(formats, key=formats.get)
    if formats[max_format] > 0:
        return max_format

    return None


def detect_column_type(series, unique_threshold=0.3, numeric_threshold=0.9):
    """
    Определяет тип данных в колонке
    """
    series_clean = series.dropna()

    if len(series_clean) == 0:
        return 'unknown'

    # Сначала проверяем boolean
    if series_clean.dtype == 'bool':
        return 'boolean'

    unique_vals = series_clean.astype(str).str.lower().str.strip().unique()

    if len(unique_vals) <= 3:
        bool_vals = {'true', 'false', '1', '0', 'yes', 'no', 'y', 'n', 't', 'f'}
        if set(unique_vals).issubset(bool_vals):
            return 'boolean'

    # Проверяем на специальные форматы
    special_format = detect_special_format(series)
    if special_format in ['email', 'phone']:
        return special_format

    # Проверяем числовые типы
    if pd.api.types.is_numeric_dtype(series):
        numeric_values = series_clean.values
        int_count = 0
        for v in numeric_values:
            if float(v).is_integer():
                int_count += 1

        int_ratio = int_count / len(numeric_values) if len(numeric_values) > 0 else 0
        return 'integer' if int_ratio >= 0.95 else 'float'

    # Пробуем преобразовать в числовой тип
    numeric_series = pd.to_numeric(series_clean, errors='coerce')
    numeric_count = numeric_series.notna().sum()
    total_count = len(series_clean)

    numeric_ratio = numeric_count / total_count if total_count > 0 else 0

    if numeric_ratio >= numeric_threshold:
        numeric_values = numeric_series.dropna().values
        int_count = 0
        for v in numeric_values:
            if float(v).is_integer():
                int_count += 1

        int_ratio = int_count / len(numeric_values) if len(numeric_values) > 0 else 0
        return 'integer' if int_ratio >= 0.95 else 'float'

    # Проверяем datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return 'datetime'

    dates = pd.to_datetime(series_clean.head(100), errors='coerce')
    if dates.notna().sum() / min(100, len(series_clean)) >= 0.8:
        return 'datetime'

    # Проверяем категориальные
    unique_vals = series_clean.astype(str).str.strip().unique()
    unique_ratio = len(unique_vals) / total_count
    if unique_ratio <= unique_threshold and len(unique_vals) < 50:
        return 'categorical'

    return 'text'


def calculate_numeric_statistics(values):
    """
    Расчет статистик для числовых данных
    """
    if len(values) == 0:
        return None

    stats_dict = {
        'mean': np.mean(values),
        'std': np.std(values) if len(values) > 1 else 0,
        'median': np.median(values),
        'min': np.min(values),
        'max': np.max(values),
        'skew': stats.skew(values) if len(values) > 2 else 0,
        'kurtosis': stats.kurtosis(values) if len(values) > 3 else 0,
        'q1': np.percentile(values, 25),
        'q3': np.percentile(values, 75),
        'n_samples': len(values)
    }

    return stats_dict


def calculate_text_statistics(entries, model):
    """
    Расчет статистик для текстовых данных
    """
    if len(entries) == 0:
        return None

    # Векторизуем данные
    vectorised_entries = vectorise_entries(entries)

    # Расчет статистик по векторам
    mean = np.mean(vectorised_entries, axis=0)
    standard_deviation = np.std(vectorised_entries, axis=0)
    median = np.median(vectorised_entries, axis=0)
    asymmetry = stats.skew(vectorised_entries, axis=0)
    excess = stats.kurtosis(vectorised_entries, axis=0)

    cov_matrix = np.cov(vectorised_entries.T)
    mean_vector = np.mean(vectorised_entries, axis=0)
    vector_dim = vectorised_entries.shape[1]

    stats_dict = {
        'overall_mean': np.mean(mean),
        'overall_std': np.mean(standard_deviation),
        'std_of_means': np.std(mean),
        'mean_of_medians': np.mean(median),
        'asymmetry_avg': np.mean(asymmetry),
        'excess_avg': np.mean(excess),
        'vector_dim': vector_dim,
        'mean_vector': mean_vector,
        'cov_matrix': cov_matrix,
        'cov_diagonal': np.diag(cov_matrix),
        'cov_trace': np.trace(cov_matrix),
        'cov_det': np.linalg.det(cov_matrix + np.eye(vector_dim) * 1e-6),
        'n_samples': len(entries),
        'vectorised_data': vectorised_entries,
        'raw_entries': entries
    }

    return stats_dict


def calculate_boolean_statistics(series):
    """
    Расчет статистик для boolean данных
    """
    series_clean = series.dropna()

    if len(series_clean) == 0:
        return None

    bool_values = []
    for val in series_clean:
        str_val = str(val).strip().lower()

        if str_val in ['true', 'yes', '1', 'y', 't'] or val is True:
            bool_values.append(True)
        elif str_val in ['false', 'no', '0', 'n', 'f'] or val is False:
            bool_values.append(False)
    if not bool_values:
        return None

    true_prob = np.mean(bool_values)

    stats_dict = {
        'true_probability': true_prob,
        'true_count': sum(bool_values),
        'false_count': len(bool_values) - sum(bool_values),
        'n_samples': len(bool_values),
        'format': detect_boolean_format(series)
    }

    return stats_dict


def calculate_categorical_statistics(series):
    """
    Расчет статистик для категориальных данных
    """
    series_clean = series.dropna()

    if len(series_clean) == 0:
        return None

    value_counts = series_clean.value_counts(normalize=True)

    stats_dict = {
        'n_categories': len(value_counts),
        'categories': value_counts.index.tolist(),
        'probabilities': value_counts.values.tolist(),
        'most_common': value_counts.index[0] if len(value_counts) > 0 else None,
        'most_common_prob': value_counts.iloc[0] if len(value_counts) > 0 else 0,
        'n_samples': len(series_clean)
    }

    return stats_dict


def calculate_datetime_statistics(series):
    """
    Расчет статистик для datetime данных
    """
    try:
        dates = pd.to_datetime(series, errors='coerce').dropna()

        if len(dates) == 0:
            return None

        # Преобразуем в Unix time (числовой формат)
        timestamps = dates.astype(np.int64) // 10 ** 9

        # Расчет статистик как для числовых данных
        numeric_stats = calculate_numeric_statistics(timestamps)

        if numeric_stats:
            # Определяем формат даты
            date_format = None
            sample_date = str(series.iloc[0]) if len(series) > 0 else ''

            if re.match(r'\d{4}-\d{2}-\d{2}', sample_date):
                date_format = 'YYYY-MM-DD'
            elif re.match(r'\d{2}/\d{2}/\d{4}', sample_date):
                date_format = 'MM/DD/YYYY'
            elif re.match(r'\d{2}-\d{2}-\d{4}', sample_date):
                date_format = 'DD-MM-YYYY'

            numeric_stats['date_format'] = date_format
            numeric_stats['min_date'] = str(dates.min())
            numeric_stats['max_date'] = str(dates.max())
            numeric_stats['timestamps'] = timestamps

        return numeric_stats
    except:
        return None


def generate_synthetic_text(stats_dict, n_samples, format_type='general_text'):
    """
    Генерация синтетических текстовых данных с сохранением статистик
    """
    if stats_dict is None:
        return [''] * n_samples

    if format_type in ['email', 'phone']:
        entries = stats_dict['raw_entries']
        if entries and len(entries) > 0:
            return np.random.choice(entries, n_samples, replace=True).tolist()
        else:
            if format_type == 'email':
                return [f"user{np.random.randint(1000, 9999)}@example.com" for _ in range(n_samples)]
            else:
                return [
                    f"+7 {np.random.randint(100, 999)} {np.random.randint(100, 999)}-{np.random.randint(10, 99)}-{np.random.randint(10, 99)}"
                    for _ in range(n_samples)]

    vector_dim = stats_dict['vector_dim']
    mean_vector = stats_dict['mean_vector']
    cov_matrix = stats_dict['cov_matrix']
    cov_matrix = cov_matrix + np.eye(vector_dim) * 1e-6

    # Генерация синтетических векторов
    try:
        synthetic_vectors = stats.multivariate_normal.rvs(
            mean=mean_vector,
            cov=cov_matrix,
            size=n_samples
        )
    except:
        synthetic_vectors = np.random.normal(
            loc=stats_dict['overall_mean'],
            scale=stats_dict['overall_std'],
            size=(n_samples, vector_dim)
        )

    # Преобразование векторов обратно в текст
    entries = []
    for vector in synthetic_vectors:
        entry = ""
        for i in range(MAX_TOKENS_PER_ENTRY):
            vectorised_token = vector[i * VECTOR_SIZE:(i + 1) * VECTOR_SIZE]
            most_similar_words = model.wv.similar_by_vector(vectorised_token, topn=3)
            for word, _ in most_similar_words:
                if str(word).isalnum():
                    entry += f" {word}"
                    break
        entries.append(entry.strip())
    return entries


def generate_synthetic_numeric(stats_dict, n_samples, is_integer=False):
    """
    Генерация синтетических числовых данных
    """
    if stats_dict is None:
        return np.zeros(n_samples)

    mean = stats_dict['mean']
    std = stats_dict['std']
    min_val = stats_dict['min']
    max_val = stats_dict['max']

    if is_integer:
        synthetic = np.random.normal(mean, std, n_samples)
        synthetic = np.clip(synthetic, min_val, max_val)
        synthetic = np.round(synthetic).astype(int)
    else:
        synthetic = np.random.normal(mean, std, n_samples)
        synthetic = np.clip(synthetic, min_val, max_val)
        synthetic = np.round(synthetic, 2)
    return synthetic


def generate_synthetic_boolean(stats_dict, n_samples, bool_format=None):
    """
    Генерация синтетических boolean данных
    """
    if stats_dict is None:
        true_prob = 0.5
    else:
        true_prob = stats_dict['true_probability']
    # Генерация boolean значений
    bool_synthetic = np.random.random(n_samples) < true_prob
    # Преобразуем в нужный формат
    if bool_format == 'true_false':
        synthetic = ['True' if x else 'False' for x in bool_synthetic]
    elif bool_format == 'yes_no':
        synthetic = ['Yes' if x else 'No' for x in bool_synthetic]
    elif bool_format == 'one_zero':
        synthetic = ['1' if x else '0' for x in bool_synthetic]
    elif bool_format == 'yn':
        synthetic = ['Y' if x else 'N' for x in bool_synthetic]
    elif bool_format == 'tf':
        synthetic = ['T' if x else 'F' for x in bool_synthetic]
    elif bool_format == 'bool':
        synthetic = bool_synthetic.astype(bool)
    else:
        synthetic = ['True' if x else 'False' for x in bool_synthetic]
    return synthetic


def generate_synthetic_categorical(stats_dict, n_samples):
    """
    Генерация синтетических категориальных данных
    """
    if stats_dict is None:
        return [''] * n_samples

    categories = stats_dict['categories']
    probabilities = stats_dict['probabilities']
    if categories and probabilities:
        return np.random.choice(categories, size=n_samples, p=probabilities).tolist()
    else:
        return [''] * n_samples


def generate_synthetic_datetime(stats_dict, n_samples, date_format=None):
    """
    Генерация синтетических datetime данных
    """
    if stats_dict is None:
        return [''] * n_samples
    mean_ts = stats_dict['mean']
    std_ts = stats_dict['std'] if stats_dict['std'] > 0 else 86400 * 30

    synthetic_ts = np.random.normal(mean_ts, std_ts, n_samples)
    synthetic_ts = np.clip(synthetic_ts, stats_dict['min'], stats_dict['max'])
    synthetic_dates = pd.to_datetime(synthetic_ts, unit='s')

    if date_format == 'YYYY-MM-DD':
        synthetic = [d.strftime('%Y-%m-%d') for d in synthetic_dates]
    elif date_format == 'MM/DD/YYYY':
        synthetic = [d.strftime('%m/%d/%Y') for d in synthetic_dates]
    elif date_format == 'DD-MM-YYYY':
        synthetic = [d.strftime('%d-%m-%Y') for d in synthetic_dates]
    else:
        synthetic = [d.strftime('%Y-%m-%d') for d in synthetic_dates]

    return synthetic

def validate_synthetic_data(original_stats, synthetic_data, column_type):
    """
    Валидация синтетических данных
    """
    validation_results = {}

    if column_type in ['text', 'email', 'phone']:
        original_vectors = original_stats['vectorised_data']
        synthetic_vectors = vectorise_entries(synthetic_data)
        orig_mean = np.mean(original_vectors, axis=0)
        synth_mean = np.mean(synthetic_vectors, axis=0)
        orig_std = np.std(original_vectors, axis=0)
        synth_std = np.std(synthetic_vectors, axis=0)
        orig_skew = stats.skew(original_vectors, axis=0)
        synth_skew = stats.skew(synthetic_vectors, axis=0)
        orig_kurt = stats.kurtosis(original_vectors, axis=0)
        synth_kurt = stats.kurtosis(synthetic_vectors, axis=0)
        validation_results = {
            'mean_error': abs(np.mean(synth_mean) - np.mean(orig_mean)),
            'std_error': abs(np.mean(synth_std) - np.mean(orig_std)),
            'skew_error': abs(np.mean(synth_skew) - np.mean(orig_skew)),
            'kurt_error': abs(np.mean(synth_kurt) - np.mean(orig_kurt)),
            'original_mean': np.mean(orig_mean),
            'original_std': np.mean(orig_std),
            'original_skew': np.mean(orig_skew),
            'original_kurt': np.mean(orig_kurt),
            'synthetic_mean': np.mean(synth_mean),
            'synthetic_std': np.mean(synth_std),
            'synthetic_skew': np.mean(synth_skew),
            'synthetic_kurt': np.mean(synth_kurt)
        }

    elif column_type in ['integer', 'float', 'datetime']:
        if column_type == 'datetime':
            original_values = original_stats['timestamps']
        else:
            original_values = original_stats.get('values', [])
        synthetic_values = synthetic_data

        if len(original_values) > 0 and len(synthetic_values) > 0:
            validation_results = {
                'mean_error': abs(np.mean(synthetic_values) - original_stats['mean']),
                'std_error': abs(np.std(synthetic_values) - original_stats['std']),
                'median_error': abs(np.median(synthetic_values) - original_stats['median']),
                'original_mean': original_stats['mean'],
                'original_std': original_stats['std'],
                'original_median': original_stats['median'],
                'synthetic_mean': np.mean(synthetic_values),
                'synthetic_std': np.std(synthetic_values),
                'synthetic_median': np.median(synthetic_values)
            }

    elif column_type == 'boolean':
        true_count = sum([1 for x in synthetic_data if str(x).lower() in ['true', 'yes', '1', 'y', 't']])
        synth_true_prob = true_count / len(synthetic_data) if len(synthetic_data) > 0 else 0
        validation_results = {
            'true_prob_error': abs(synth_true_prob - original_stats['true_probability']),
            'original_true_prob': original_stats['true_probability'],
            'synthetic_true_prob': synth_true_prob,
            'original_true_count': original_stats['true_count'],
            'original_false_count': original_stats['false_count'],
            'synthetic_true_count': true_count,
            'synthetic_false_count': len(synthetic_data) - true_count
        }

    elif column_type == 'categorical':
        synth_value_counts = pd.Series(synthetic_data).value_counts(normalize=True)
        validation_results = {
            'n_categories_error': abs(len(synth_value_counts) - original_stats['n_categories']),
            'original_n_categories': original_stats['n_categories'],
            'synthetic_n_categories': len(synth_value_counts),
            'most_common_match': synth_value_counts.index[0] == original_stats['most_common'] if len(
                synth_value_counts) > 0 else False
        }

    return validation_results


def visualize_distributions(original_stats, synthetic_data, column_name, column_type, n_dimensions=5):
    """
    Визуализация распределений
    """
    if column_type in ['text', 'email', 'phone']:
        original_vectors = original_stats['vectorised_data']
        synthetic_vectors = vectorise_entries(synthetic_data)

        col_name = column_name
        dim_indices = np.linspace(0, original_vectors.shape[1] - 1, n_dimensions, dtype=int)

        fig, axes = plt.subplots(2, n_dimensions, figsize=(4 * n_dimensions, 8))
        fig.suptitle(f'Распределения по измерениям: {col_name}', fontsize=14, fontweight='bold')

        for i, dim_idx in enumerate(dim_indices):
            axes[0, i].hist(original_vectors[:, dim_idx], bins=50, alpha=0.7, color='blue',
                            label='Исходные', density=True)
            axes[0, i].set_title(f'Измерение {dim_idx}')
            axes[0, i].set_xlabel('Значение')
            axes[0, i].set_ylabel('Плотность')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)

            axes[1, i].hist(synthetic_vectors[:, dim_idx], bins=50, alpha=0.7, color='red',
                            label='Синтетические', density=True)
            axes[1, i].set_title(f'Измерение {dim_idx}')
            axes[1, i].set_xlabel('Значение')
            axes[1, i].set_ylabel('Плотность')
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        n_compare = min(3, n_dimensions)
        fig, axes = plt.subplots(1, n_compare, figsize=(5 * n_compare, 4))
        if n_compare == 1:
            axes = [axes]
        fig.suptitle(f'Сравнение распределений: {col_name}', fontsize=14, fontweight='bold')

        for i, dim_idx in enumerate(dim_indices[:n_compare]):
            axes[i].hist(original_vectors[:, dim_idx], bins=50, alpha=0.6, color='blue',
                         label='Исходные', density=True)
            axes[i].hist(synthetic_vectors[:, dim_idx], bins=50, alpha=0.6, color='red',
                         label='Синтетические', density=True)
            axes[i].set_title(f'Измерение {dim_idx}')
            axes[i].set_xlabel('Значение')
            axes[i].set_ylabel('Плотность')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        if original_vectors.shape[1] > 2:
            combined = np.vstack([original_vectors, synthetic_vectors])
            pca = PCA(n_components=2)
            pca.fit(combined)
            orig_2d = pca.transform(original_vectors)
            synth_2d = pca.transform(synthetic_vectors)

            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.scatter(orig_2d[:, 0], orig_2d[:, 1], alpha=0.5, s=10,
                       c='blue', label='Исходные', marker='o')
            ax.scatter(synth_2d[:, 0], synth_2d[:, 1], alpha=0.5, s=10,
                       c='red', label='Синтетические', marker='^')
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            ax.set_title(f'2D проекция через PCA: {col_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

    elif column_type in ['integer', 'float']:
        original_values = original_stats.get('values', [])
        synthetic_values = synthetic_data

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Распределение значений: {column_name}', fontsize=14, fontweight='bold')
        axes[0].hist(original_values, bins=50, alpha=0.7, color='blue', label='Исходные', density=True)
        axes[0].set_title('Исходные данные')
        axes[0].set_xlabel('Значение')
        axes[0].set_ylabel('Плотность')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[1].hist(synthetic_values, bins=50, alpha=0.7, color='red', label='Синтетические', density=True)
        axes[1].set_title('Синтетические данные')
        axes[1].set_xlabel('Значение')
        axes[1].set_ylabel('Плотность')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.hist(original_values, bins=50, alpha=0.6, color='blue', label='Исходные', density=True)
        ax.hist(synthetic_values, bins=50, alpha=0.6, color='red', label='Синтетические', density=True)
        ax.set_title(f'Сравнение распределений: {column_name}')
        ax.set_xlabel('Значение')
        ax.set_ylabel('Плотность')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    elif column_type == 'boolean':
        original_true = original_stats['true_count']
        original_false = original_stats['false_count']

        synthetic_true = sum([1 for x in synthetic_data if str(x).lower() in ['true', 'yes', '1', 'y', 't']])
        synthetic_false = len(synthetic_data) - synthetic_true

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f'Распределение boolean значений: {column_name}', fontsize=14, fontweight='bold')

        axes[0].bar(['True', 'False'], [original_true, original_false], color=['green', 'red'], alpha=0.7)
        axes[0].set_title('Исходные данные')
        axes[0].set_ylabel('Количество')
        axes[0].grid(True, alpha=0.3)

        axes[1].bar(['True', 'False'], [synthetic_true, synthetic_false], color=['green', 'red'], alpha=0.7)
        axes[1].set_title('Синтетические данные')
        axes[1].set_ylabel('Количество')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    elif column_type == 'categorical':
        original_categories = original_stats['categories']
        original_probs = original_stats['probabilities']

        synth_value_counts = pd.Series(synthetic_data).value_counts(normalize=True)
        synth_categories = synth_value_counts.index.tolist()
        synth_probs = synth_value_counts.values.tolist()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Распределение категорий: {column_name}', fontsize=14, fontweight='bold')

        axes[0].bar(range(len(original_categories[:10])), original_probs[:10], alpha=0.7, color='blue')
        axes[0].set_title('Исходные данные (топ-10)')
        axes[0].set_xlabel('Категория')
        axes[0].set_ylabel('Вероятность')
        axes[0].set_xticks(range(len(original_categories[:10])))
        axes[0].set_xticklabels(original_categories[:10], rotation=45, ha='right')
        axes[0].grid(True, alpha=0.3)

        axes[1].bar(range(len(synth_categories[:10])), synth_probs[:10], alpha=0.7, color='red')
        axes[1].set_title('Синтетические данные (топ-10)')
        axes[1].set_xlabel('Категория')
        axes[1].set_ylabel('Вероятность')
        axes[1].set_xticks(range(len(synth_categories[:10])))
        axes[1].set_xticklabels(synth_categories[:10], rotation=45, ha='right')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

def main():
    global df_sample
    SAMPLE_SIZE = min(1000, len(df))
    df_sample = df.sample(SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    column_stats = {}
    column_types = {}
    column_formats = {}

    for column in df_sample.columns:
        col_type = detect_column_type(df_sample[column])
        column_types[column] = col_type
        if col_type in ['text', 'email', 'phone']:
            entries = df_sample[column].dropna().astype(str).tolist()
            stats_dict = calculate_text_statistics(entries, model)
            column_stats[column] = stats_dict
            print(f"  Текстовых записей: {len(entries)}")
            print(f"  Размерность векторов: {stats_dict['vector_dim'] if stats_dict else 'N/A'}")

        elif col_type in ['integer', 'float']:
            values = pd.to_numeric(df_sample[column], errors='coerce').dropna().values
            stats_dict = calculate_numeric_statistics(values)
            if stats_dict:
                stats_dict['values'] = values
            column_stats[column] = stats_dict
            print(f"  Числовых значений: {len(values)}")
            print(
                f"  Диапазон: {stats_dict['min'] if stats_dict else 'N/A'} - {stats_dict['max'] if stats_dict else 'N/A'}")

        elif col_type == 'boolean':
            stats_dict = calculate_boolean_statistics(df_sample[column])
            column_formats[column] = stats_dict['format'] if stats_dict and 'format' in stats_dict else None
            column_stats[column] = stats_dict
            if stats_dict:
                print(f"  Boolean значений: {stats_dict['n_samples']}")
                print(f"  Вероятность True: {stats_dict['true_probability']:.2f}")
                print(f"  Формат: {stats_dict.get('format', 'N/A')}")

        elif col_type == 'categorical':
            stats_dict = calculate_categorical_statistics(df_sample[column])
            column_stats[column] = stats_dict
            if stats_dict:
                print(f"  Категориальных значений: {stats_dict['n_samples']}")
                print(f"  Количество категорий: {stats_dict['n_categories']}")
                print(f"  Самая частая категория: {stats_dict['most_common']}")

        elif col_type == 'datetime':
            stats_dict = calculate_datetime_statistics(df_sample[column])
            column_formats[column] = stats_dict['date_format'] if stats_dict and 'date_format' in stats_dict else None
            column_stats[column] = stats_dict
            if stats_dict:
                print(f"  Дата/время значений: {stats_dict['n_samples']}")
                print(f"  Диапазон дат: {stats_dict['min_date']} - {stats_dict['max_date']}")
                print(f"  Формат даты: {stats_dict.get('date_format', 'N/A')}")

    n_samples = min(500, SAMPLE_SIZE)
    synthetic_data = {}
    validation_results = {}

    for column, col_type in column_types.items():
        stats_dict = column_stats.get(column)

        if col_type in ['text', 'email', 'phone']:
            format_type = 'email' if col_type == 'email' else 'phone' if col_type == 'phone' else 'general_text'
            synthetic = generate_synthetic_text(stats_dict, n_samples, format_type)
        elif col_type == 'integer':
            synthetic = generate_synthetic_numeric(stats_dict, n_samples, is_integer=True)
        elif col_type == 'float':
            synthetic = generate_synthetic_numeric(stats_dict, n_samples, is_integer=False)
        elif col_type == 'boolean':
            bool_format = column_formats.get(column)
            synthetic = generate_synthetic_boolean(stats_dict, n_samples, bool_format)
        elif col_type == 'categorical':
            synthetic = generate_synthetic_categorical(stats_dict, n_samples)
        elif col_type == 'datetime':
            date_format = column_formats.get(column)
            synthetic = generate_synthetic_datetime(stats_dict, n_samples, date_format)
        else:
            synthetic = [''] * n_samples
        synthetic_data[column] = synthetic

        # Валидация
        if stats_dict:
            validation = validate_synthetic_data(stats_dict, synthetic, col_type)
            validation_results[column] = validation

    synthetic_df = pd.DataFrame(synthetic_data)
    synthetic_df.to_csv('synthetic_data_enhanced.csv', index=False)
    for column, validation in validation_results.items():
        if validation:
            print(f"\n{column} ({column_types[column]}):")
            for key, value in validation.items():
                print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
   # Выбираем по одной колонке каждого типа для визуализации
    types_to_visualize = ['text', 'integer', 'boolean', 'categorical', 'datetime']
    visualized = {}

    for col_type in types_to_visualize:
        cols_of_type = [col for col, t in column_types.items() if t == col_type]
        if cols_of_type:
            column_to_visualize = cols_of_type[0]
            if column_to_visualize in column_stats and column_to_visualize in synthetic_data:
                print(f"\nВизуализация для колонки: {column_to_visualize} ({col_type})")
                visualize_distributions(
                    column_stats[column_to_visualize],
                    synthetic_data[column_to_visualize],
                    column_to_visualize,
                    col_type
                )
                visualized[col_type] = column_to_visualize

    type_summary = {}
    for col_type in column_types.values():
        type_summary[col_type] = type_summary.get(col_type, 0) + 1

    for col_type, count in type_summary.items():
        example_col = next((col for col, t in column_types.items() if t == col_type), None)
        print(f"\n{col_type.upper()}: {count} колонок")
        if example_col:
            print(f"  Пример: {example_col}")
            if example_col in validation_results:
                errors = [v for k, v in validation_results[example_col].items() if 'error' in k]
                if errors:
                    avg_error = np.mean(errors)
                    print(f"  Средняя ошибка валидации: {avg_error:.4f}")

    return column_stats, synthetic_df, validation_results


# Запуск
if __name__ == "__main__":
    column_stats, synthetic_df, validation_results = main()