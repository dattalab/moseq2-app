import click
from pathlib import Path
from moseq2_app.steps import train_flip_classifier
from moseq2_common.options import ConfigFileError
from moseq2_common.project import OldProjectError


@click.command()
@click.argument("data_index_path", type=click.Path(exists=True))
@click.option("--prefix", default="", type=str, help="Prefix for classifier file name")
@click.option(
    "--classifier-type",
    default="svm",
    type=click.Choice(["svm", "rf"]),
    help="Classifier model to use. Either a support vector machine (svm) or random forest (rf)",
)
@click.option(
    "--config-file",
    type=click.Path(),
    default=None,
    help="Path to the config.yaml you extract with: each session's training frames are cleaned with its "
         "extract section and the session's session_config.yaml entry; extraction's defaults without it",
)
def main(data_index_path, prefix, classifier_type, config_file):
    if len(prefix) > 0 and not prefix.endswith("_"):
        prefix += "_"

    print(f"Training {classifier_type} classifier")
    try:
        # each session cleaned as extraction cleans its recording, or with extraction's defaults
        train_flip_classifier(
            data_index_path,
            Path(data_index_path).with_name(f"{prefix}{classifier_type}_flip_classifier.p"),
            config_file,
            classifier=classifier_type.upper(),
        )
    except (ConfigFileError, OldProjectError) as error:
        raise click.ClickException(str(error)) from None


if __name__ == "__main__":
    main()
