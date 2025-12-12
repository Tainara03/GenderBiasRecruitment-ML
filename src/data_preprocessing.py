import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_dataset(path: str) -> pd.DataFrame:
    """
    Carrega o dataset a partir do caminho informado.
    """
    df = pd.read_csv(path)
    return df

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpa nomes de colunas:
    - Remove parênteses
    - Substitui espaços por underscore
    - Substitui vírgulas por underscore
    - Mantém hífens
    """
    df = df.copy()
    df.columns = [
        col.strip()
           .replace(" ", "")
           .replace("(", "_")
           .replace(")", "")
           .replace(",", "_")
        for col in df.columns
    ]
    return df


def encode_categorical(df: pd.DataFrame, categorical_cols: list = None) -> pd.DataFrame:
    """
    Converte colunas categóricas para numéricas usando LabelEncoder.
    """
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include="object").columns.tolist()
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
    return df, le_dict

def split_features_target(df: pd.DataFrame, target: str):
    """
    Separa features e variável alvo.
    """
    X = df.drop(columns=[target])
    y = df[target]
    return X, y

def train_test_split_stratified(X, y, test_size=0.3, random_state=42):
    """
    Divide em treino e teste mantendo proporção da variável alvo.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
