# This script migrates qdrant collection from one environment to another
# Environment can be same qdrant instance or different qdrant instance
"""
How to run:

python -m backend.migration.qdrant_migration \
--source_backend_url http://localhost:8000 \
--source_qdrant_url https://ps-qdrant-prathamesh-ws.devtest.truefoundry.tech \
--source_collection_name creditcard \
--destination_backend_url http://localhost:8000 \
--destination_qdrant_url https://ps-qdrant-prathamesh-ws.devtest.truefoundry.tech \
--destination_collection_name creditcopy

Add --source_prefix & --destination_prefix if qdrant instance has prefix enabled
Add --batch_size to set batch size for migration
Add --overwrite to overwrite destination collection if exists in separate qdrant
"""


import argparse

import requests
from qdrant_client import QdrantClient

from backend.logger import logger
from backend.migration.utils import get_collection, migrate
from backend.types import AssociateDataSourceWithCollection, CreateCollectionDto


def migrate_collection(
    source_backend_url: str,
    source_collection_name: str,
    source_qdrant_url: str,
    source_prefix: str,
    destination_backend_url: str,
    destination_collection_name: str,
    destination_qdrant_url: str,
    destination_prefix: str,
    batch_size: int,
    overwrite: bool,
):
    try:
        fetched_source_collection = get_collection(
            source_backend_url, source_collection_name, type="source"
        )
    except Exception as e:
        raise e

    logger.debug(f"Source collection found '{fetched_source_collection.get('name')}' ")

    same_qdrant_loc = (source_qdrant_url == destination_qdrant_url) and (
        source_prefix == destination_prefix
    )

    try:
        fetched_destination_collection = get_collection(
            destination_backend_url, destination_collection_name, type="destination"
        )
        if fetched_destination_collection and same_qdrant_loc:
            raise Exception(
                f"Source and destination qdrant locations are same. Destination collection '{destination_collection_name}' already exists."
            )

        elif fetched_destination_collection and not overwrite and not same_qdrant_loc:
            raise Exception(
                f"Source and destination qdrant locations are different. Destination collection '{destination_collection_name}' already exists in the destination qdrant. Add --overwrite to overwrite the collection."
            )
        else:
            logger.debug(
                f"Destination collection '{destination_collection_name}' not found at destination. Proceeding..."
            )
    except Exception as e:
        raise e

    dest_collection = None
    try:
        # prepare collection to be created at destination
        dest_collection = CreateCollectionDto(
            name=destination_collection_name,
            description=fetched_source_collection.get(
                "description", f"Collection cloned from {source_collection_name}"
            ),
            embedder_config=fetched_source_collection.get("embedder_config"),
            associated_data_sources=[
                AssociateDataSourceWithCollection(
                    data_source_fqn=value.get("data_source_fqn"),
                    parser_config=value.get("parser_config"),
                )
                for _, value in fetched_source_collection.get(
                    "associated_data_sources"
                ).items()
            ],
        ).model_dump()

        logger.debug(
            f"Creating '{dest_collection.get('name')}' collection at destination"
        )
        # create collection at destination
        with requests.post(
            url=f"{destination_backend_url.rstrip('/')}/v1/collections/",
            json=dest_collection,
        ) as r:
            r.raise_for_status()
            logger.debug("Collection entry created: ", r.json())

        logger.debug("Collection migration started...")

        source_qdrant_client = QdrantClient(
            url=source_qdrant_url,
            port=443 if source_qdrant_url.startswith("https://") else 6333,
            prefer_grpc=False if source_qdrant_url.startswith("https://") else True,
            prefix=source_prefix,
        )

        destination_qdrant_client = QdrantClient(
            url=destination_qdrant_url,
            port=443 if destination_qdrant_url.startswith("https://") else 6333,
            prefer_grpc=(
                False if destination_qdrant_url.startswith("https://") else True
            ),
            prefix=destination_prefix,
        )

        migrate(
            source_client=source_qdrant_client,
            dest_client=destination_qdrant_client,
            source_collection_name=source_collection_name,
            destination_collection_name=destination_collection_name,
            batch_size=batch_size,
            same_qdrant=same_qdrant_loc,
        )

    except Exception as e:
        if dest_collection is not None:
            with requests.delete(
                url=f"{destination_backend_url.rstrip('/')}/v1/collections/{destination_collection_name}",
                json=dest_collection,
            ) as r:
                logger.debug("Destination collection entry deleted: ", r.json())
        raise e

    logger.debug(
        f"Collection '{source_collection_name}' migrated to '{destination_collection_name}' successfully..."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Migrate qdrant collection from one environment to another"
    )

    # source
    parser.add_argument(
        "--source_backend_url", type=str, help="Source backend url", required=True
    )
    parser.add_argument(
        "--source_collection_name",
        type=str,
        help="Source collection name",
        required=True,
    )
    parser.add_argument(
        "--source_qdrant_url", type=str, help="Source qdrant url", required=True
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        help="Source qdrant prefix",
        required=False,
        default=None,
    )

    # destination
    parser.add_argument(
        "--destination_backend_url",
        type=str,
        help="Destination backend url",
        required=True,
    )
    parser.add_argument(
        "--destination_collection_name",
        type=str,
        help="Destination collection name",
        required=True,
    )
    parser.add_argument(
        "--destination_qdrant_url",
        type=str,
        help="Destination qdrant url",
        required=True,
    )
    parser.add_argument(
        "--destination_prefix",
        type=str,
        help="Destination qdrant prefix",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for migration",
        required=False,
        default=100,
    )
    parser.add_argument(
        "--overwrite",
        help="Overwrite destination collection if exists in separate qdrant",
        required=False,
        action="store_true",
    )

    args = parser.parse_args()

    migrate_collection(
        # source args
        source_backend_url=args.source_backend_url,
        source_collection_name=args.source_collection_name,
        source_qdrant_url=args.source_qdrant_url,
        source_prefix=args.source_prefix,
        # destination args
        destination_backend_url=args.destination_backend_url,
        destination_collection_name=args.destination_collection_name,
        destination_qdrant_url=args.destination_qdrant_url,
        destination_prefix=args.destination_prefix,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
