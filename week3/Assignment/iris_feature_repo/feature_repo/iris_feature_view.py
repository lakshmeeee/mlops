from datetime import timedelta
from feast import Entity, FeatureView, Field, ValueType
from feast.types import Float32, String
from feast.infra.offline_stores.bigquery_source import BigQuerySource

iris_bq_source = BigQuerySource(
    table="sheetgpt-385916.iris_dataset.iris_table",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

iris_entity = Entity(name="iris_id", join_keys=["iris_id"], value_type=ValueType.INT64,)

iris_feature_view = FeatureView(
    name="iris_features",
    entities=[iris_entity],
    ttl=timedelta(days=1),
    schema=[
        Field(name="sepal_length", dtype=Float32),
        Field(name="sepal_width", dtype=Float32),
        Field(name="petal_length", dtype=Float32),
        Field(name="petal_width", dtype=Float32),
        Field(name="species", dtype=String),
    ],
    source=iris_bq_source,
)
