"""Submit dynamic epoch-1 Kaggle notebook and surface API errors."""

from kaggle.api.kaggle_api_extended import KaggleApi


api = KaggleApi()
api.authenticate()
try:
    result = api.competition_submit_code(
        "submission.csv",
        "Raptor v5 ResNet50 fusion raw epoch 1, minimum-overlap 384 V2, dynamic test inference",
        "rsna-knee-abnormality-detection",
        "zafarnematov/rsna-raptor-resnet50-epoch1-submit",
        3,
    )
    print(result)
except Exception as error:
    print(f"{type(error).__name__}: {error}")
    response = getattr(error, "response", None)
    if response is not None:
        print(f"HTTP status: {response.status_code}")
        print(f"Kaggle response: {response.text[:4000]}")
    raise
