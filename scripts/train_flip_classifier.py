import click
from pathlib import Path
from moseq2_app.steps import train_flip_classifier
from moseq2_common.frames import CleaningOptions


@click.command()
@click.argument("data_index_path", type=click.Path(exists=True))
@click.option("--prefix", default="", type=str, help="Prefix for classifier file name")
@click.option(
    "--classifier-type",
    default="svm",
    type=click.Choice(["svm", "rf"]),
    help="Classifier model to use. Either a support vector machine (svm) or random forest (rf)",
)
def main(data_index_path, prefix, classifier_type):
    if len(prefix) > 0 and not prefix.endswith("_"):
        prefix += "_"

    print(f"Training {classifier_type} classifier")
    # cleaned with extraction's default cleaning settings
    train_flip_classifier(
        data_index_path,
        Path(data_index_path).with_name(f"{prefix}{classifier_type}_flip_classifier.p"),
        CleaningOptions(),
        classifier=classifier_type.upper(),
    )


if __name__ == "__main__":
    main()
