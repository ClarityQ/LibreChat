import datetime
from enum import Enum
from enum import StrEnum
from typing import List, Optional, Tuple, Dict, Any, Annotated, Union
from typing import Literal, Type, TypeVar, cast
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict

T = TypeVar("T", bound=Enum)


def enum_to_literal(enum_class: Type[T]) -> Type:
    values = tuple(e.value for e in enum_class)
    return cast(Type, Literal[values])  # type: ignore


class ElementType(StrEnum):
    NODE = "NODE"
    EDGE = "EDGE"


class NodeType(StrEnum):
    ENTITY = "ENTITY"
    HIERARCHY = "HIERARCHY"
    FEATURE = "FEATURE"
    DIMENSION = "DIMENSION"
    METRIC = "METRIC"
    SEGMENT = "SEGMENT"
    FUNNEL = "FUNNEL"
    TIME = "TIME"
    RECIPE = "RECIPE"


NodeTypeLiteral = enum_to_literal(enum_class=NodeType)


class RelationshipType(StrEnum):
    SCOPES = "SCOPES"
    SCOPED_BY = "SCOPED_BY"
    SEGMENTS = "SEGMENTS"
    SEGMENTED_BY = "SEGMENTED_BY"
    FOLLOWS = "FOLLOWS"
    FOLLOWED_BY = "FOLLOWED_BY"
    MEASURES = "MEASURES"
    MEASURED_WITH = "MEASURED_BY"
    TIMES = "TIMES"
    TIMED_BY = "TIMED_BY"
    SAME_AS = "SAME_AS"
    CONTAINS = "CONTAINS"
    CONTAINED_BY = "CONTAINED_BY"


class EdgeProperties(BaseModel):
    source: str
    target: str

    model_config = ConfigDict(extra="allow")


class ComponentProperties(BaseModel):
    name: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    data_sample: Optional[str] = None
    sql: Optional[str] = None
    sql_table: Optional[str] = None
    aliases: Optional[List[str]] = None
    data_source: Optional[str] = None
    tags: Optional[List[str]] = None
    meta: Optional[Dict[str, Any]] = None
    active: Optional[bool] = True

    model_config = ConfigDict(extra="allow")


class SemanticComponent(BaseModel):
    semantic_type: str
    element_type: ElementType
    properties: ComponentProperties = Field(default_factory=ComponentProperties)


class SemanticNode(SemanticComponent):
    element_type: ElementType = ElementType.NODE


class DimensionType(StrEnum):
    TIME = "time"
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    GEO = "geo"


class DimensionFormat(StrEnum):
    ID = "id"
    URL = "url"
    CURRENCY = "currency"
    PERCENT = "percent"


class DimensionGranularity(BaseModel):
    name: str
    interval: str
    offset: Optional[str] = None
    origin: Optional[str] = None
    title: Optional[str] = None


class DimensionProperties(ComponentProperties):
    primary_key: Optional[bool] = None
    type: Optional[DimensionType] = None
    case: Optional[List[Tuple[str, str]]] = None
    format: Optional[DimensionFormat] = None
    granularity: Optional[List[DimensionGranularity]] = None


class Dimension(SemanticNode):
    semantic_type: Literal["DIMENSION"] = "DIMENSION"
    properties: DimensionProperties = Field(default_factory=DimensionProperties)


class RecipeProperties(ComponentProperties):
    pass


class Recipe(SemanticNode):
    semantic_type: Literal["RECIPE"] = "RECIPE"
    properties: RecipeProperties = Field(default_factory=RecipeProperties)


class EntityProperties(ComponentProperties):
    foreign_keys: Optional[Dict[str, str]] = None
    dimensions: Optional[Dict[str, Dimension]] = None
    recipes: Optional[Dict[str, Recipe]] = None


class Entity(SemanticNode):
    semantic_type: Literal["ENTITY"] = "ENTITY"
    properties: EntityProperties = Field(default_factory=EntityProperties)


class Hierarchy(SemanticNode):
    semantic_type: Literal["HIERARCHY"] = "HIERARCHY"


class FeatureProperties(ComponentProperties):
    foreign_keys: Optional[Dict[str, str]] = None
    dimensions: Optional[Dict[str, Dimension]] = None
    recipes: Optional[Dict[str, Recipe]] = None


class Feature(SemanticNode):
    semantic_type: Literal["FEATURE"] = "FEATURE"
    properties: FeatureProperties = Field(default_factory=FeatureProperties)


class MetricType(StrEnum):
    STRING = "string"
    TIME = "time"
    BOOLEAN = "boolean"
    NUMBER = "number"
    COUNT = "count"
    COUNT_DISTINCT = "count_distinct"
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"


class MetricFormat(StrEnum):
    PERCENT = "percent"
    CURRENCY = "currency"


class MetricProperties(ComponentProperties):
    type: Optional[MetricType] = None
    format: Optional[MetricFormat] = None
    filters: Optional[List["str"]] = None
    recipes: Optional[Dict[str, Recipe]] = None


class Metric(SemanticNode):
    semantic_type: Literal["METRIC"] = "METRIC"
    properties: MetricProperties = Field(default_factory=MetricProperties)


class Segment(SemanticNode):
    semantic_type: Literal["SEGMENT"] = "SEGMENT"


class FunnelStep(BaseModel):
    name: str
    description: Optional[str] = None
    sql: Optional[str] = None
    required: Optional[bool] = True
    order: Optional[int] = None


class FunnelProperties(ComponentProperties):
    steps: List[FunnelStep] = Field(default_factory=list)
    conversion_window: Optional[str] = None
    strict_sequence: Optional[bool] = True


class Funnel(SemanticNode):
    semantic_type: Literal["FUNNEL"] = "FUNNEL"
    properties: FunnelProperties = Field(default_factory=FunnelProperties)


class TimeNode(SemanticNode):
    semantic_type: Literal["TIME"] = "TIME"


ComponentTypes = Annotated[
    Union[
        Entity,
        Hierarchy,
        Feature,
        Metric,
        Segment,
        Funnel,
        Recipe,
    ],
    Field(discriminator="semantic_type"),
]


class SemanticEdge(SemanticComponent):
    element_type: ElementType = ElementType.EDGE
    semantic_type: RelationshipType
    properties: EdgeProperties

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SemanticComponentUpdate(BaseModel):
    properties: Optional[Dict[str, Any]] = None


class SemanticFabricResponse(SemanticComponent):
    model_config = ConfigDict(from_attributes=True)

    id: str
    created_at: datetime.datetime
    last_updated: datetime.datetime
    product_id: UUID


class SemanticFabricFilters(BaseModel):
    query: Optional[str] = None
    semantic_type: Optional[List[str]] = None
    element_type: Optional[List[ElementType]] = None
    node_types: Optional[List[NodeTypeLiteral]] = None
    relationship_types: Optional[List[RelationshipType]] = None
    active: Optional[bool] = None
    created_after: Optional[datetime.datetime] = None
    created_before: Optional[datetime.datetime] = None
    component_name: Optional[str] = None
    dimension_name: Optional[str] = None
    recipe_name: Optional[str] = None
    fuzzy_search: Optional[bool] = Field(default=False, description="Use trigram similarity for fuzzy matching")
    similarity_threshold: Optional[float] = Field(
        default=0.3, description="Minimum similarity threshold for fuzzy search (0.0-1.0)"
    )
    match_threshold: Optional[float] = Field(
        default=1.0, description="Minimum percentage of words to match (0.0-1.0). Example: 0.75 for a 3/4 match"
    )


class SemanticFabricSearchQuery(BaseModel):
    skip: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1)
    sort_order: str = Field(default="asc", pattern="^(asc|desc)$")
    sort_by: str = Field(default="name")
    filters: SemanticFabricFilters = Field(default_factory=SemanticFabricFilters)


class ComponentByNameRequest(BaseModel):
    name: str


class BulkCreateResponse(BaseModel):
    total: int
    created: int
    updated: int
    skipped: int
    errors: List[str] = Field(default_factory=list)


class ImportRequest(BaseModel):
    components: List[ComponentTypes] = Field(default_factory=list)
    replace_existing: bool = False


class ExportResponse(BaseModel):
    components: List[SemanticFabricResponse] = Field(default_factory=list)
    export_timestamp: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.UTC))


class UpdateOperation(StrEnum):
    SET = "SET"
    ADD = "ADD"
    REMOVE = "REMOVE"


class ComponentPropertiesUpdate(BaseModel):
    """Base properties that all component types can update."""

    semantic_type: str

    title: Optional[str] = None
    description: Optional[str] = None
    data_sample: Optional[str] = None
    sql: Optional[str] = None
    sql_table: Optional[str] = None
    aliases: Optional[List[str]] = None
    data_source: Optional[str] = None
    tags: Optional[List[str]] = None
    meta: Optional[Dict[str, Any]] = None
    active: Optional[bool] = None


class EntityPropertiesUpdate(ComponentPropertiesUpdate):
    """Entity-specific properties that can be updated."""

    semantic_type: Literal["ENTITY"] = "ENTITY"

    foreign_keys: Optional[Dict[str, str]] = None


class FeaturePropertiesUpdate(ComponentPropertiesUpdate):
    """Feature-specific properties that can be updated."""

    semantic_type: Literal["FEATURE"] = "FEATURE"

    foreign_keys: Optional[Dict[str, str]] = None


class MetricPropertiesUpdate(ComponentPropertiesUpdate):
    """Metric-specific properties that can be updated."""

    semantic_type: Literal["METRIC"] = "METRIC"

    type: Optional[MetricType] = None
    format: Optional[MetricFormat] = None
    filters: Optional[List[str]] = None


class SegmentPropertiesUpdate(ComponentPropertiesUpdate):
    """Segment-specific properties that can be updated."""

    semantic_type: Literal["SEGMENT"] = "SEGMENT"


class FunnelPropertiesUpdate(ComponentPropertiesUpdate):
    """Funnel-specific properties that can be updated."""

    semantic_type: Literal["FUNNEL"] = "FUNNEL"

    steps: Optional[List[Dict[str, Any]]] = None
    conversion_window: Optional[str] = None
    strict_sequence: Optional[bool] = None


class HierarchyPropertiesUpdate(ComponentPropertiesUpdate):
    """Hierarchy-specific properties that can be updated."""

    semantic_type: Literal["HIERARCHY"] = "HIERARCHY"


class RecipePropertiesUpdate(ComponentPropertiesUpdate):
    """Recipe-specific properties that can be updated."""

    semantic_type: Literal["RECIPE"] = "RECIPE"


class DimensionPropertiesUpdate(ComponentPropertiesUpdate):
    """Dimension-specific properties that can be updated."""

    semantic_type: Literal["DIMENSION"] = "DIMENSION"

    primary_key: Optional[bool] = None
    type: Optional[DimensionType] = None
    format: Optional[DimensionFormat] = None
    granularity: Optional[List[DimensionGranularity]] = None


ComponentOrDimension = Union[ComponentTypes, Dimension]
PropertiesUpdateTypes = Annotated[
    Union[
        EntityPropertiesUpdate,
        FeaturePropertiesUpdate,
        MetricPropertiesUpdate,
        SegmentPropertiesUpdate,
        FunnelPropertiesUpdate,
        HierarchyPropertiesUpdate,
        DimensionPropertiesUpdate,
        RecipePropertiesUpdate,
    ],
    Field(discriminator="semantic_type"),
]
