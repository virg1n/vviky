import click

from baseten.common import api


@click.group()
def pretrained():
    """Explore pre-trained model available on Baseten."""
    pass


@pretrained.command()
@click.option("--all", is_flag=True, default=False)
def list(all):
    """Lists all pretrained models"""
    pretty_print_pretrained(api.pretrained_models()["pretrained_models"], all)


@pretrained.command()
@click.argument("pretrained_model_name")
def get(pretrained_model_name):
    """Get pretrained model, given zoo name"""
    pretty_print_pretrained(
        [api.get_pretrained_model(pretrained_model_name)["pretrained_model"]], all
    )


def pretty_print_pretrained(models, show_all=True):
    for model in models:
        available = False
        if model["s3_key"]:
            available = True
        elif not show_all:
            continue
        click.echo(f"Name: {model['pretty_name']} | Key: {model['name']} | Available: {available}")
