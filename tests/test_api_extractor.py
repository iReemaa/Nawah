from app.ingestion import (
    validate_api_url,
    APIExtractorError
)


def test_invalid_api_url():

    try:
        validate_api_url(
            "not-a-valid-url"
        )

    except APIExtractorError:
        print(
            "Invalid API URL test passed."
        )

    else:
        raise AssertionError(
            "Invalid API URL should have failed."
        )


if __name__ == "__main__":
    test_invalid_api_url()