import asyncio
import json
from typing import Optional
from uuid import UUID

from loguru import logger
from mcp.server.fastmcp import FastMCP

from clarityq_mcp.semantic_fabric_client import (
    SemanticFabricClient,
    format_client_error,
    MemoryCreate,
)
from clarityq_mcp.semantic_fabric_common import (
    render_data_model_overview,
    render_memory,
    render_search_results,
    apply_patch_by_operation_type,
    view_model,
    view_dimension,
    view_recipe,
    delete_model,
    delete_dimension,
    delete_recipe,
)
from clarityq_mcp.semantic_fabric_config import PRODUCT_ID, DATA_WAREHOUSE_TYPE
from clarityq_mcp.semantic_fabric_models import (
    SemanticFabricFilters,
    SemanticFabricSearchQuery,
    NodeTypeLiteral,
    SemanticComponentUpdate,
    Dimension,
    Recipe,
    ComponentTypes,
    UpdateOperation,
    PropertiesUpdateTypes,
)

mcp = FastMCP("Semantic Fabric")


@mcp.tool()
async def view_component(component_path: str) -> str:
    """View details of any semantic component using a structured path syntax.

    ## When to Use
    - View model details: `"models/<model_name>"` (e.g., "models/users")
    - View dimension details: `"models/<model_name>/dimensions/<dimension_name>"` (e.g., "models/users/dimensions/user_id")
    - View recipe details: `"models/<model_name>/recipes/<recipe_name>"` (e.g., "models/users/recipes/user_report")
    - When exploring existing components or verifying properties before updates

    ## When NOT to Use
    - For listing all models (use list_models instead)
    - For searching models by pattern (use search_models instead)
    - For modifying components (use update_* tools instead)

    ## Path Format Rules
    - Paths are case-sensitive and must match exact component names
    - Include spaces as-is (e.g., "models/user activity/dimensions/session time")
    - Always follow the exact structure: models → dimensions/recipes → name

    ## Examples

    <example>
    User: What properties does the users model have?
    Assistant: *Uses view_component("models/users")*
    Assistant: The users model is an ENTITY type with SQL table "public.users" and dimensions: user_id, email, signup_date
    </example>

    <example>
    User: How is the signup_date dimension configured?
    Assistant: *Uses view_component("models/users/dimensions/signup_date")*
    Assistant: The signup_date dimension is type: time, format: date, with SQL: DATE(created_at)
    </example>

    <example>
    User: How can I generate reports for individual users?
    Assistant: *First searches models, then uses view_component("models/users/recipes/user_report")*
    Assistant: I found a user_report recipe that provides a template with SQL: SELECT * FROM user_activities WHERE user_id = {user_id}
    </example>
    """
    # Parse the component path
    path_parts = component_path.split("/")

    # Check if the path follows the expected format
    if len(path_parts) >= 2 and path_parts[0] == "models":
        # It starts with "models/" as required

        if len(path_parts) == 2:
            # This is a model path: "models/<model_name>"
            model_name = path_parts[1]
            return await view_model(model_name)
        elif len(path_parts) == 4 and path_parts[2] == "dimensions":
            # This is a dimension path: "models/<model_name>/dimensions/<dimension_name>"
            model_name = path_parts[1]
            dimension_name = path_parts[3]
            return await view_dimension(model_name, dimension_name)
        elif len(path_parts) == 4 and path_parts[2] == "recipes":
            # This is a recipe path: "models/<model_name>/recipes/<recipe_name>"
            model_name = path_parts[1]
            recipe_name = path_parts[3]
            return await view_recipe(model_name, recipe_name)
        else:
            # Invalid format
            return (
                f"Invalid component path: '{component_path}'. Path must be in format 'models/<model_name>', "
                f"'models/<model_name>/dimensions/<dimension_name>', or 'models/<model_name>/recipes/<recipe_name>'."
            )
    else:
        # Missing "models/" prefix
        return (
            f"Invalid component path: '{component_path}'. Path must start with 'models/' and be in format 'models/<model_name>', "
            f"'models/<model_name>/dimensions/<dimension_name>', or 'models/<model_name>/recipes/<recipe_name>'."
        )


@mcp.tool()
async def delete_component(component_path: str) -> str:
    """Use this tool to delete any semantic component using a structured path syntax.

    ## IMPORTANT: Only use this tool when a user explicitly requests deletion

    Only use this tool when the user has explicitly asked you to delete or remove a component. Never
    proactively suggest deletion or use this tool unless specifically requested.

    ## Path Syntax and Format

    1. **Standard Path Format**:
       - For models: `"models/<model_name>"`
       - For dimensions: `"models/<model_name>/dimensions/<dimension_name>"`
       - For recipes: `"models/<model_name>/recipes/<recipe_name>"`

    2. **Path Examples**:
       - `"models/users"` → Deletes the entire users model
       - `"models/users/dimensions/email"` → Deletes only the email dimension from the users model
       - `"models/users/recipes/user_report"` → Deletes only the user_report recipe from the users model

    ## Examples of Use

    <example>
    User: Please delete the test_users model I created earlier.
    Assistant: I'll delete the test_users model for you.
    *Uses delete_component("models/test_users") to remove the model*
    Assistant: The test_users model has been successfully deleted.
    </example>

    <example>
    User: I need to remove the signup_date dimension in the users model so I can recreate it correctly.
    Assistant: I'll remove the signup_date dimension from the users model.
    *Uses delete_component("models/users/dimensions/signup_date") to remove the dimension*
    Assistant: The signup_date dimension has been successfully removed from the users model.
    </example>

    ## WARNING

    Deletion operations cannot be undone. Always verify that you are deleting the correct component
    and that it's not referenced by other components before proceeding.
    """
    # Parse the component path
    path_parts = component_path.split("/")

    # Check if the path follows the expected format
    if len(path_parts) >= 2 and path_parts[0] == "models":
        # It starts with "models/" as required

        if len(path_parts) == 2:
            # This is a model path: "models/<model_name>"
            model_name = path_parts[1]
            return await delete_model(model_name)
        elif len(path_parts) == 4 and path_parts[2] == "dimensions":
            # This is a dimension path: "models/<model_name>/dimensions/<dimension_name>"
            model_name = path_parts[1]
            dimension_name = path_parts[3]
            return await delete_dimension(model_name, dimension_name)
        elif len(path_parts) == 4 and path_parts[2] == "recipes":
            # This is a recipe path: "models/<model_name>/recipes/<recipe_name>"
            model_name = path_parts[1]
            recipe_name = path_parts[3]
            return await delete_recipe(model_name, recipe_name)
        else:
            # Invalid format
            return (
                f"Invalid component path: '{component_path}'. Path must be in format 'models/<model_name>', "
                f"'models/<model_name>/dimensions/<dimension_name>', or 'models/<model_name>/recipes/<recipe_name>'."
            )
    else:
        # Missing "models/" prefix
        return (
            f"Invalid component path: '{component_path}'. Path must start with 'models/' and be in format 'models/<model_name>', "
            f"'models/<model_name>/dimensions/<dimension_name>', or 'models/<model_name>/recipes/<recipe_name>'."
        )


@mcp.tool()
async def create_model(model: ComponentTypes) -> str:
    """Creates a new semantic model of the specified type.

    This tool returns a JSON template for the specified model type
    that you can modify via update_model

    Args:
        model: Model definition object
    """
    client = SemanticFabricClient()
    try:
        await client.create_component(product_id=UUID(PRODUCT_ID), component=model)
        return f"Successfully created model '{model.properties.name}'"
    except Exception as e:
        return f"Error creating model: {str(e)}"


@mcp.tool()
async def update_model(
    model_name: str,
    properties: PropertiesUpdateTypes,
    operation: Optional[UpdateOperation] = UpdateOperation.SET,
) -> str:
    """Update a semantic model using different update operations.

    This tool updates a model using the specified operation type:
    - SET: Directly set values, replacing existing ones
    - ADD: Merge new values with existing ones (extending lists, merging objects)
    - REMOVE: Remove specified values or keys

    The properties parameter should be one of these Pydantic models:
    - EntityPropertiesUpdate: For ENTITY models
      Example: {"description": "Updated user entity", "foreign_keys": {"user_id": "users.id"}}
    - FeaturePropertiesUpdate: For FEATURE models
      Example: {"description": "Login event", "sql": "SELECT * FROM logins", "sql_table": "logins"}
    - MetricPropertiesUpdate: For METRIC models
      Example: {"description": "Revenue metric", "type": "SUM", "format": "CURRENCY"}
    - SegmentPropertiesUpdate: For SEGMENT models
      Example: {"description": "Active users", "sql": "WHERE last_login > NOW() - INTERVAL '30 days'"}
    - FunnelPropertiesUpdate: For FUNNEL models
      Example: {"description": "Purchase funnel", "steps": [{"name": "view", "order": 1}, {"name": "purchase", "order": 2}]}
    - HierarchyPropertiesUpdate: For HIERARCHY models
      Example: {"description": "Product category hierarchy", "meta": {"levels": 3}}
    - ComponentPropertiesUpdate: For generic updates
      Example: {"description": "Updated description", "tags": ["important"]}

    Args:
        model_name: Name of the model to update
        properties: A Pydantic model containing the properties to update
        operation: The update operation type (SET, ADD, REMOVE)

    Returns:
        Confirmation of the update
    """
    client = SemanticFabricClient()

    current_model = await client.get_component_by_name(product_id=UUID(PRODUCT_ID), name=model_name)
    if not current_model:
        return f"Model not found with name: {model_name}"

    try:
        operation = operation or UpdateOperation.SET
        update_operation = UpdateOperation(operation.upper())
    except ValueError:
        return f"Invalid operation: {operation}. Must be one of: SET, ADD, REMOVE"

    updated_properties = apply_patch_by_operation_type(
        operation=update_operation,
        current_model=current_model,
        update_properties=properties,
    )

    component_update = SemanticComponentUpdate(properties=updated_properties)

    logger.debug(f"Updated component: {component_update.model_dump()}")

    result = await client.update_component_by_name(
        product_id=UUID(PRODUCT_ID),
        component_name=model_name,
        component=component_update,
        dimension_name=None,
    )

    if not result:
        return f"Failed to update model '{model_name}'"

    return f"Successfully updated model '{model_name}'"


@mcp.tool()
async def create_dimension(
    model_name: str,
    dimension: Dimension,
) -> str:
    """Creates a new dimension within an entity or feature model.

    This tool adds a dimension to an existing entity or feature model.
    Dimensions provide attributes that can be used for grouping and filtering.

    Args:
        model_name: Name of the entity or feature model to add the dimension to
        dimension: A DimensionNode object defining the dimension properties

    Returns:
        Confirmation of the dimension creation with the dimension structure
    """
    client = SemanticFabricClient()

    # First check if the model exists and is an entity or feature
    component = await client.get_component_by_name(product_id=UUID(PRODUCT_ID), name=model_name)
    if not component:
        return f"Model not found with name: {model_name}"

    # Check if model is an entity or feature
    if component.semantic_type not in ["ENTITY", "FEATURE"]:
        return f"Dimensions can only be added to ENTITY or FEATURE models, but '{model_name}' is a {component.semantic_type} model"

    # Get existing properties
    existing_properties = dict(component.properties)
    dimensions = existing_properties.get("dimensions", {}) or {}

    # Check if dimension already exists
    if dimension.properties.name in dimensions:
        return f"Dimension '{dimension.properties.name}' already exists in model '{model_name}'"

    # Add the new dimension
    dimensions[dimension.properties.name] = dimension.model_dump()

    # Update the model with the new dimension
    component_update = SemanticComponentUpdate(properties={"dimensions": dimensions})
    updated = await client.update_component_by_name(
        product_id=UUID(PRODUCT_ID),
        component_name=model_name,
        dimension_name=None,
        component=component_update,
    )

    if not updated:
        return f"Failed to update model '{model_name}' with new dimension"

    return f"Successfully created dimension '{dimension.properties.name}' in model '{model_name}'"


@mcp.tool()
async def update_dimension(
    model_name: str,
    dimension_name: str,
    properties: PropertiesUpdateTypes,
    operation: Optional[UpdateOperation] = UpdateOperation.SET,
) -> str:
    """Update a dimension using different update operations.

    This tool updates a dimension using the specified operation type:
    - SET: Directly set values, replacing existing ones
    - ADD: Merge new values with existing ones (extending lists, merging objects)
    - REMOVE: Remove specified values or keys

    The properties parameter should be a DimensionPropertiesUpdate Pydantic model.

    Examples:
    1. Update dimension type and format:
       {"type": "STRING", "format": "ID", "description": "User ID dimension"}

    2. Update time dimension granularity:
       {"type": "TIME", "granularity": [{"name": "daily", "interval": "1d"}]}

    3. Add tags without changing other properties:
       {"tags": ["important", "filter"]} with operation="ADD"

    4. Remove specific tags:
       {"tags": ["deprecated"]} with operation="REMOVE"

    Args:
        model_name: Name of the entity or feature model containing the dimension
        dimension_name: Name of the dimension to update
        properties: DimensionPropertiesUpdate Pydantic model with properties to update
        operation: The update operation type (SET, ADD, REMOVE)

    Returns:
        Confirmation of the update
    """
    client = SemanticFabricClient()

    component = await client.get_component_by_name(product_id=UUID(PRODUCT_ID), name=model_name)
    if not component:
        return f"Model not found with name: {model_name}"

    # Check if model has dimensions
    if not hasattr(component.properties, "dimensions") or not component.properties.dimensions:
        return f"Model '{model_name}' doesn't have any dimensions"

    # Check if dimension exists
    if dimension_name not in component.properties.dimensions:
        return f"Dimension '{dimension_name}' not found in model '{model_name}'"

    try:
        operation = operation or UpdateOperation.SET
        update_operation = UpdateOperation(operation.upper())
    except ValueError:
        return f"Invalid operation: {operation}. Must be one of: SET, ADD, REMOVE"

    dimension_data = component.properties.dimensions[dimension_name]
    dimension = Dimension.model_validate(dimension_data)

    # Apply the patch using the operation type
    updated_properties = apply_patch_by_operation_type(
        operation=update_operation,
        current_model=dimension,
        update_properties=properties,
    )

    # Create component update for API
    component_update = SemanticComponentUpdate(properties=updated_properties)

    # Update using existing API
    result = await client.update_component_by_name(
        product_id=UUID(PRODUCT_ID),
        component_name=model_name,
        component=component_update,
        dimension_name=dimension_name,
    )

    if not result:
        return f"Failed to update dimension '{dimension_name}' in model '{model_name}'"

    return f"Successfully updated dimension '{dimension_name}' in model '{model_name}'"


@mcp.tool()
async def create_recipe(
    model_name: str,
    recipe: Recipe,
) -> str:
    """Creates a new recipe within a model.

    This tool adds a recipe to an existing entity, feature, or metric model.
    Recipes provide reusable query templates or data transformation logic.

    Args:
        model_name: Name of the model to add the recipe to
        recipe: A Recipe object defining the recipe properties

    Returns:
        Confirmation of the recipe creation with the recipe structure
    """
    client = SemanticFabricClient()

    # First check if the model exists and is an appropriate type
    component = await client.get_component_by_name(product_id=UUID(PRODUCT_ID), name=model_name)
    if not component:
        return f"Model not found with name: {model_name}"

    # Check if model is of an appropriate type
    if component.semantic_type not in ["ENTITY", "FEATURE", "METRIC"]:
        return f"Recipes can only be added to ENTITY, FEATURE, or METRIC models, but '{model_name}' is a {component.semantic_type} model"

    # Get existing properties
    existing_properties = dict(component.properties)
    recipes = existing_properties.get("recipes", {}) or {}

    # Check if recipe already exists
    if recipe.properties.name in recipes:
        return f"Recipe '{recipe.properties.name}' already exists in model '{model_name}'"

    # Add the new recipe
    recipes[recipe.properties.name] = recipe.model_dump()

    # Update the model with the new recipe
    component_update = SemanticComponentUpdate(properties={"recipes": recipes})
    updated = await client.update_component_by_name(
        product_id=UUID(PRODUCT_ID),
        component_name=model_name,
        dimension_name=None,
        recipe_name=None,
        component=component_update,
    )

    if not updated:
        return f"Failed to update model '{model_name}' with new recipe"

    return f"Successfully created recipe '{recipe.properties.name}' in model '{model_name}'"


@mcp.tool()
async def update_recipe(
    model_name: str,
    recipe_name: str,
    properties: PropertiesUpdateTypes,
    operation: Optional[UpdateOperation] = UpdateOperation.SET,
) -> str:
    """Update a recipe using different update operations.

    This tool updates a recipe using the specified operation type:
    - SET: Directly set values, replacing existing ones
    - ADD: Merge new values with existing ones (extending lists, merging objects)
    - REMOVE: Remove specified values or keys

    The properties parameter should be a RecipePropertiesUpdate Pydantic model.

    Examples:
    1. Update recipe description and SQL:
       {"description": "Updated user report template", "sql": "SELECT * FROM users WHERE region = {region}"}

    2. Add tags without changing other properties:
       {"tags": ["important", "reporting"]} with operation="ADD"

    3. Remove specific tags:
       {"tags": ["deprecated"]} with operation="REMOVE"

    Args:
        model_name: Name of the model containing the recipe
        recipe_name: Name of the recipe to update
        properties: RecipePropertiesUpdate Pydantic model with properties to update
        operation: The update operation type (SET, ADD, REMOVE)

    Returns:
        Confirmation of the update
    """
    client = SemanticFabricClient()

    component = await client.get_component_by_name(product_id=UUID(PRODUCT_ID), name=model_name)
    if not component:
        return f"Model not found with name: {model_name}"

    # Check if model has recipes
    if not hasattr(component.properties, "recipes") or not component.properties.recipes:
        return f"Model '{model_name}' doesn't have any recipes"

    # Check if recipe exists
    if recipe_name not in component.properties.recipes:
        return f"Recipe '{recipe_name}' not found in model '{model_name}'"

    try:
        operation = operation or UpdateOperation.SET
        update_operation = UpdateOperation(operation.upper())
    except ValueError:
        return f"Invalid operation: {operation}. Must be one of: SET, ADD, REMOVE"

    recipe_data = component.properties.recipes[recipe_name]
    recipe = Recipe.model_validate(recipe_data)

    # Apply the patch using the operation type
    updated_properties = apply_patch_by_operation_type(
        operation=update_operation,
        current_model=recipe,
        update_properties=properties,
    )

    # Create component update for API
    component_update = SemanticComponentUpdate(properties=updated_properties)

    # Update using existing API
    result = await client.update_component_by_name(
        product_id=UUID(PRODUCT_ID),
        component_name=model_name,
        component=component_update,
        dimension_name=None,
        recipe_name=recipe_name,
    )

    if not result:
        return f"Failed to update recipe '{recipe_name}' in model '{model_name}'"

    return f"Successfully updated recipe '{recipe_name}' in model '{model_name}'"


@mcp.tool()
async def list_models() -> str:
    """Lists all semantic models in the data layer organized by type.

    Returns a hierarchical tree view of available models (Entities, Features, Metrics, etc.).
    For Entity and Feature models, it shows their available dimensions as nested elements.
    Use this to get a quick overview of the semantic layer before working with specific models.

    Returns:
        Tree view of available semantic models with their dimensions
    """
    try:
        client = SemanticFabricClient()
        components = await client.get_all_components(product_id=UUID(PRODUCT_ID))

        if not components:
            return "No models found for this product"

        overview = render_data_model_overview(components)
        system_notice = (
            '<clarityq_system_notice status="internal" visibility="system_only">\n'
            "      - Note that the names shown here are display titles. The actual SQL implementation\n"
            '        can be found in the "sql" field when viewing detailed information. Always check\n'
            "        the SQL implementation to understand the underlying data.\n"
            '      - To view detailed model information: view_component("models/model_name")\n'
            '      - To view dimension details: view_component("models/model_name/dimensions/dimension_name")\n'
            '      - To view recipe details: view_component("models/model_name/recipes/recipe_name")\n'
            "</clarityq_system_reminder>"
        )
        return overview + "\n\n" + system_notice
    except Exception as e:
        return format_client_error(e, context="listing models")


@mcp.tool()
async def search_models(
    search_text: str = "",
    type_filter: Optional[NodeTypeLiteral] = None,
    limit: int = 50,
    sort_by: str = "rank",
) -> str:
    """Search for semantic models across the entire semantic layer.

    ## When to Use
    - Discover models without knowing exact names
    - Find models related to specific concepts (e.g., "revenue", "user")
    - Filter models by type (e.g., all METRICS)
    - Look for models containing specific text in any property
    - Create targeted lists instead of viewing the entire catalog

    ## When NOT to Use
    - When you know the exact model name (use view_component instead)
    - When you want a complete overview of all models (use list_models instead)
    - When you need detailed info about a specific model (use view_component instead)

    ## Search Features
    - **Multi-term search**: Space-separated terms are joined with OR logic
      Example: `"user profile"`
    - **Type filtering**: Narrow results to specific model types
      Example: `search_models("", type_filter="METRIC")` → lists all metrics

    ## Examples

    <example>
    User: Find all models related to revenue data.
    Assistant: *Uses search_models("revenue")*
    Assistant: Found models: METRIC/monthly_revenue, FEATURE/revenue_events, ENTITY/revenue_source
    </example>

    <example>
    User: What metrics are available in the system?
    Assistant: *Uses search_models("", type_filter="METRIC")*
    Assistant: Found these metrics: daily_active_users, monthly_revenue, conversion_rate...
    </example>

    <example>
    User: Find models for analyzing both users and sessions.
    Assistant: *Uses search_models("users|sessions")*
    Assistant: Found models related to users and sessions: users, user_activity, sessions...
    </example>
    """
    try:
        client = SemanticFabricClient()
        semantic_type_filter = [type_filter] if type_filter is not None else None
        filters = SemanticFabricFilters(query=search_text, semantic_type=semantic_type_filter, match_threshold=0.67)
        search_query = SemanticFabricSearchQuery(filters=filters, skip=0, limit=limit, sort_by=sort_by)
        components, total = await client.search_components(product_id=UUID(PRODUCT_ID), query=search_query)
        return render_search_results(components, search_text=search_text)
    except Exception as e:
        context = "searching models"
        return format_client_error(e, context=context)


@mcp.tool()
async def execute_sql(query: str) -> str:
    """Execute a SQL query against the data warehouse (READ ONLY).

    This tool allows running SQL queries against the data warehouse and returns the results.
    It is READ ONLY and cannot modify any data.

    Args:
        query: The SQL query to execute (SELECT statements only)

    Returns:
        The query results as formatted JSON
    """
    try:
        client = SemanticFabricClient()
        result = await client.execute_sql_query(query=query, product_id=PRODUCT_ID)
        return json.dumps(result)
    except Exception as e:
        return format_client_error(e, context="executing SQL")


@mcp.tool()
async def batch_execute(operations: list[dict]) -> dict:
    """Execute multiple operations in parallel on the Semantic Fabric server.

    ## When to Use
    - Run multiple independent operations efficiently in a single call
    - Retrieve several components at once (e.g., multiple models or dimensions)
    - Perform coordinated updates across multiple components
    - Create complex structures involving multiple related components
    - Reduce conversation overhead for multiple related operations

    ## When NOT to Use
    - For single operations (use the specific tool directly)
    - For sequential operations where later steps depend on earlier results
    - When operations must stop on first error
    - For very long-running operations that might timeout when combined

    ## Operation Format
    Each operation must be a dictionary with:
    - `tool`: Name of the tool to call (string)
    - `arguments`: Arguments to pass to the tool (object)

    ```json
    {
      "tool": "view_component",
      "arguments": {
        "component_path": "models/users"
      }
    }
    ```

    ## Examples

    <example>
    User: I need details about the users model and its email dimension.
    Assistant: *Uses batch_execute([
        {"tool": "view_component", "arguments": {"component_path": "models/users"}},
        {"tool": "view_component", "arguments": {"component_path": "models/users/dimensions/email"}}
    ])*
    Assistant: The users model is an ENTITY with an email dimension of type STRING...
    </example>

    <example>
    User: Create a user_activity model with login_time and session_duration dimensions.
    Assistant: *Uses batch_execute to create model and dimensions in one operation*
    Assistant: I've created the user_activity model with the login_time and session_duration dimensions.
    </example>

    ## Return Structure
    Returns a dictionary mapping operation indices to their results:
    ```json
    {
      "op_0": {"result": "Result of first operation"},
      "op_1": {"result": "Result of second operation"},
      "op_2": {"error": "Error message if operation failed"}
    }
    ```

    Note: Operations run in parallel. If one fails, others continue executing.
    """
    results = {}
    tasks = []

    # Create tasks for all operations
    for i, op in enumerate(operations):
        tool_name = op.get("tool")
        arguments = op.get("arguments", {})

        # Use the mcp internal context to access the tool
        context = mcp.get_context()

        try:
            # Create task using the internal tool manager's call_tool method
            task = asyncio.create_task(mcp._tool_manager.call_tool(tool_name, arguments, context))
            tasks.append((i, task))
        except Exception as e:
            results[f"op_{i}"] = {"error": f"Tool '{tool_name}' not found or error in setup: {str(e)}"}
            continue

    # Wait for all tasks to complete
    for i, task in tasks:
        try:
            result = await task
            results[f"op_{i}"] = {"result": str(result)}
        except Exception as e:
            results[f"op_{i}"] = {"error": str(e)}

    return results


@mcp.tool()
async def think(thought: str) -> str:
    """Use this tool to think about something without changing any data.

    This tool allows you to record your reasoning process during analysis.
    It does not obtain new information or change the database,
    but just appends the thought to the log for tracking your analytical process.
    Use it when complex reasoning or some cache memory is needed.

    Args:
        thought: A thought to think about and record

    Returns:
        Confirmation that the thought was recorded
    """
    logger.debug(f"THOUGHT: {thought}")
    return "Thought recorded"


@mcp.tool()
async def memory_edit(old_content: Optional[str], new_content: str, expected_replacements: int = 1) -> str:
    """Edit the content in the memory.

    This tool allows you to modify the memory content using a find-and-replace approach.
    It looks for an exact match of old_content within the memory and replaces it with new_content.

    If memory doesn't exist yet, it will be created with new_content.

    ## IMPORTANT: Only use this tool when a user explicitly requests to modify, update, or create memory.
    Never proactively suggest using this tool. Memory contains critical configuration data that should
    only be modified when directly requested.

    To make a memory edit, provide the following:
    1. old_content: The text to replace (must match exactly, including whitespace)
    2. new_content: The text that will replace old_content
    3. expected_replacements: The number of replacements you expect to make (default: 1)

    Special operations:
    - To create new memory or append to existing: Use "" or None for old_content and provide your content

    CRITICAL REQUIREMENTS FOR USING THIS TOOL:

    1. UNIQUENESS (when expected_replacements is not specified): The old_content MUST uniquely identify the specific instance you want to change:
       - Include sufficient context before and after the change point (at least 3-5 lines or enough text to ensure uniqueness)
       - Include all whitespace and formatting exactly as it appears
       - Ensure the string appears only once in memory

    2. EXPECTED MATCHES: If you want to replace multiple instances:
       - Use expected_replacements with the exact number of occurrences
       - If the actual number of matches doesn't equal expected_replacements, the edit will fail
       - This validation prevents unintended changes

    3. VERIFICATION: Before using this tool:
       - Check how many instances of the target text exist in memory
       - If multiple instances exist, either:
         a) Gather enough context to uniquely identify each one and make separate calls, OR
         b) Use expected_replacements parameter with the exact count of instances you expect to replace

    Args:
        old_content: The text to replace (exact match required)
        new_content: The text that will replace old_content
        expected_replacements: Number of occurrences that must exist for validation (default: 1)

    Returns:
        Confirmation of the edit operation
    """
    client = SemanticFabricClient()

    try:
        # Try to get existing memory
        memory = await client.get_memory(product_id=UUID(PRODUCT_ID))

        # Case 1: No memory exists yet
        if not memory:
            if not new_content:
                return "No memory exists and no content provided to create one."
            # Create new memory with the provided content
            new_memory = MemoryCreate(content=new_content)
            await client.set_memory(product_id=UUID(PRODUCT_ID), memory=new_memory)
            return "Memory created successfully."

        # Case 2: Empty or None old_content - append to existing memory
        if old_content is None or old_content == "":
            # For empty old_content or None, append to existing memory
            existing_content = memory.content.rstrip()
            append_content = new_content.lstrip()
            # Add exactly one empty line between existing content and new content
            updated_content = f"{existing_content}\n\n{append_content}"
            new_memory = MemoryCreate(content=updated_content)
            await client.set_memory(product_id=UUID(PRODUCT_ID), memory=new_memory)
            return "Content appended to memory successfully."

        # Case 3: Standard find-and-replace operation
        occurrences = memory.content.count(old_content)

        # Validate the expected number of replacements
        if occurrences != expected_replacements:
            return (
                f"Error: Found {occurrences} matches of the string to replace, "
                f"but expected {expected_replacements}. The number of actual matches must equal "
                f"the expected replacements. Please adjust your string to match or update the expected count."
            )

        # Perform the replacement of all occurrences (since we verified the count)
        updated_content = memory.content.replace(old_content, new_content)

        # Save the updated content
        new_memory = MemoryCreate(content=updated_content)
        await client.set_memory(product_id=UUID(PRODUCT_ID), memory=new_memory)

        # Return appropriate message based on number of replacements
        if expected_replacements == 1:
            return "Memory content updated successfully (1 replacement)."
        else:
            return f"Memory content updated successfully ({expected_replacements} replacements)."

    except Exception as e:
        return format_client_error(e, context="editing memory")


@mcp.prompt()
async def build_semantic_layer_guide() -> str:
    """Guide the model to lead an interactive semantic layer building session."""

    client = SemanticFabricClient()

    # First, load memory content if it exists
    memory_content = ""
    memory = await client.get_memory(product_id=UUID(PRODUCT_ID))
    if memory:
        memory_content = render_memory(memory)

    # Then, load semantic data model
    model_overview = ""
    components = await client.get_all_components(product_id=UUID(PRODUCT_ID))
    if components:
        model_overview = render_data_model_overview(components)

    prompt = f"""
    You are an expert semantic modeling assistant. Your role is to guide the user through building a comprehensive semantic data model layer for their business domain. This is a collaborative process where you will take the lead in asking questions, suggesting components, and utilizing the appropriate tools to create the semantic fabric.
    data_warehouse_type: {DATA_WAREHOUSE_TYPE}
    
    ## Tone and Communication Style

    Be concise, direct, and focused. When communicating:
    - Keep explanations brief and practical
    - Use clear, structured formats like lists and tables
    - Provide specific examples rather than general theory
    - Answer questions directly without unnecessary preamble
    - Present options clearly with concrete next steps
    - Balance being helpful with being efficient

    ## Proactiveness and Balanced Guidance

    Be proactive, but only when appropriate. Strike a careful balance between:

    1. Taking initiative when the user asks for guidance on semantic modeling
    2. Executing exactly what the user asks without adding unwanted suggestions 
    3. Providing helpful next steps without overwhelming with too many options

    When a user asks about something specific ("How do I create a dimension?"), answer their question directly first before suggesting actions. When they ask for general guidance ("Help me model my e-commerce data"), take more initiative in driving the process.

    Effective proactive guidance looks like:
    - After creating a model: "Your Users entity is now created. Would you like to add some dimensions like location or user_type?"
    - After viewing a model: "I see this model lacks a primary key. Should we add one?"
    - When the user seems uncertain: "Based on what you've described, I'd recommend starting with these three entities: Users, Products, and Orders. Which would you like to build first?"

    After completing tasks:
    - Don't provide explanation summaries unless asked
    - Simply confirm success and ask about next steps
    - Avoid lengthy explanations of work you just completed
    - Focus on keeping the modeling process moving forward

    Remember that you're a collaborative assistant - guide when needed, follow when directed.

    ## Error Handling

    When errors occur:
    - Share the exact error message clearly and concisely
    - Wait for user guidance before attempting fixes
    - For API errors, focus on the most relevant parts of the message
    - Suggest targeted troubleshooting steps when appropriate
    - After resolving, briefly confirm success and continue the workflow

    For data-related issues:
    - Verify data availability with sample SQL queries
    - Check for mismatches between model definitions and actual data
    - Focus on solving one error completely before moving on

    ## Task Execution Approach

    When building the semantic layer, follow this efficient process:

    1. Understand requirements through focused questions
    2. Explore existing components before creating new ones
    3. Plan model structure with clear entity relationships
    4. Implement components systematically, verifying each step
    5. Refine based on feedback and real-world usage needs

    This isn't a rigid workflow - adapt to the user's priorities while ensuring all necessary components are properly connected. Implement the most valuable components first to deliver immediate analytical value.

    ## SQL Style and Best Practices

    When writing or suggesting SQL:
    - ALWAYS write standalone, fully qualified queries that can be executed independently
    - Use complete schema and table references (e.g., "public.users" not just "users")
    - Include all required columns and joins for the query to work without external context
    - Follow {DATA_WAREHOUSE_TYPE} syntax precisely
    - Keep statements readable with proper indentation and line breaks
    - Use consistent capitalization (keywords in UPPER case, identifiers in snake_case)
    - Include descriptive comments for complex logic
    - Use explicit column names instead of SELECT *
    - Apply appropriate LIMIT clauses for data exploration

    For SQL queries requiring dynamic values (dates, IDs, user inputs):
    - Use clear placeholder variables with descriptive names wrapped in curly braces
    - Add comments explaining what each placeholder represents
    - Provide example values to show the expected format
    - Never hardcode specific values that would limit the query's reusability

    ## Available Tools
    
    ### Discovery Tools
    - `list_models(type_filter)`: View the current semantic models organized by type
    - `search_models(search_text, type_filter, limit, sort_by)`: Find models using flexible search criteria
      - Use wildcards (like "User*") for name pattern matching
      - Use plain text (like "revenue") to search across all model properties
    
    ### Viewing Tools
    - `view_component("models/<model_name>")`: Get details of a specific model
    - `view_component("models/<model_name>/dimensions/<dimension_name>")`: Get details of a specific dimension
    - `view_component("models/<model_name>/recipes/<recipe_name>")`: Get details of a specific recipe
    
    ### Creation Tools
    - `create_model(model_type, name)`: Create a new model (ENTITY, FEATURE, METRIC, etc.)
    - `create_dimension(model_name, dimension)`: Add a dimension to an entity or feature
    - `create_recipe(model_name, recipe)`: Add a recipe to an entity, feature, or metric model
    
    ### Update Tools
    - `update_model(model_name, properties, operation)`: Update a model with specified properties
    - `update_dimension(model_name, dimension_name, properties, operation)`: Update a dimension with specified properties
    - `update_recipe(model_name, recipe_name, properties, operation)`: Update a recipe with specified properties
    
    ### Deletion Tools
    - `delete_model(model_name)`: Delete a model
    - `delete_dimension(model_name, dimension_name)`: Delete a dimension
    - `delete_recipe(model_name, recipe_name)`: Delete a recipe
    
    ### Data Access Tools
    - `execute_sql(query)`: Run a READ ONLY SQL query against the {DATA_WAREHOUSE_TYPE} data warehouse

    ## Semantic Model Components
    | Type | Examples | Definition |
    |------|----------|------------|
    | **ENTITY** | Users, Organizations, Products | Core business objects that are the primary subjects in your data system. They have unique identifiers, can have dimensions, and provide context for metrics and grouping. |
    | **FEATURE** | Comments, Purchases, Views | Specific actions, events, or measurable attributes of entities. They store raw data points that will be aggregated into metrics and can have dimensions for context. |
    | **DIMENSION** | Country, Device Type, Plan Tier, Time | Contextual attributes used for filtering, grouping, and segmentation. They add analytical context to entities and features, enabling data analysis from different perspectives. |
    | **RECIPE** | User Report, Revenue By Region, Active Subscriptions | Pre-defined semantic queries and transformations associated with models. They provide reusable, parameterized query patterns for common analysis needs. |
    | **METRIC** | ARPU, Total Revenue, User Count | Calculation methods applied to features to produce business KPIs. They use aggregation (sum, count, average) on features to transform raw data into meaningful measurements. |
    | **SEGMENT** | Premium Users, New Customers | Filtered subsets of entities based on specific criteria. They create logical groupings for targeted analysis and comparison between different groups. |
    | **FUNNEL** | Purchase Flow, Onboarding | Sequential user journeys with defined steps. They track progression through a series of actions, measure conversion rates, and identify drop-off points. |

    ## Implementation Workflow
    
    1. **Domain Discovery**:
       - Ask about business domain, key entities, and analytical goals
       - Explore available data sources with sample SQL queries
       - Identify top 3-5 business questions to be answered

    2. **Foundation Layer**:
       - Build core ENTITY models with proper properties and keys
       - Add essential DIMENSIONS to each entity
       - Test entities by checking their properties with view_model()

    3. **Behavioral Layer**:
       - Define FEATURES for measurable activities
       - Add dimensions to features for analysis context

    4. **Analytical Layer**:
       - Create METRICS to calculate business KPIs
       - Define SEGMENTS for filtered analysis
       - Create FUNNELS for sequential journey analysis

    5. **Validation & Refinement**:
       - Test models with exploratory SQL
       - Verify component properties
       - Refine based on feedback
       - Document all models thoroughly
       
    ## Semantic Layer Structure
    
    When exploring the semantic layer with tools like `list_models()`, you'll be working with a hierarchical structure that follows this general pattern:
    
    ```yaml
    CATEGORY_TYPE:                        # Top-level category (ENTITY, FEATURE, METRIC, etc.)
      ModelName:                          # Specific model instance
        dimensions:                       # Container for dimensions (if any)
        - dimension_name_1                # Individual dimension
        - dimension_name_2                # Another dimension
        recipes:                          # Container for recipes (if any)
        - recipe_name_1                   # Individual recipe
        - recipe_name_2                   # Another recipe
      AnotherModel: {{}}                    # Model with no dimensions or recipes (empty)
    ANOTHER_CATEGORY:                     # Another top-level category
      SomeModel:                          # Another model instance
        dimensions:
        - some_dimension
        recipes:
        - some_recipe
        # More dimensions and recipes may be listed...
      EmptyModel: {{}}                      # Another model with no dimensions or recipes
    # Additional categories and models...
    ```
    
    This structure represents the semantic fabric where:
    - Top-level entries are categories representing different model types
    - Second-level entries are individual models within each category
    - Third-level entries are dimensions and recipes associated with each model
    - Empty brackets `{{}}` indicate models without dimensions or recipes
    
    To navigate and explore this structure effectively:
    1. Start with `list_models()` to see the complete overview
    2. Use `view_component("models/model_name")` to examine specific model details
    3. Use `view_component("models/model_name/dimensions/dimension_name")` for dimension details
    4. Use `view_component("models/model_name/recipes/recipe_name")` for recipe details
    5. Use search functionality with `search_models()` when it's not obvious which models you need to examine

    """

    contexts = ""
    if memory_content:
        contexts += "\n" + memory_content + "\n"
    if model_overview:
        contexts += "\n" + model_overview

    prompt += "\n" + contexts

    return prompt


@mcp.prompt()
async def product_analytics_assistant() -> str:
    """Guide the model to act as a product analyst helping with analytics questions."""
    import textwrap

    client = SemanticFabricClient()

    # First, load memory content if it exists
    memory_content = ""
    memory = await client.get_memory(product_id=UUID(PRODUCT_ID))
    if memory:
        memory_content = render_memory(memory)

    # Then, load semantic data model
    components = await client.get_all_components(product_id=UUID(PRODUCT_ID))
    model_overview = ""
    if components:
        model_overview = render_data_model_overview(components)

    prompt = f"""
    # Product Analytics Assistant

    You are an expert product analyst assisting with business questions using the semantic data model. Translate analytical needs to data queries, execute them, and provide clear insights with visualizations.
    
    **Data Warehouse**: {DATA_WAREHOUSE_TYPE}
    
    ## Core Principles
    
    1. **Data Integrity**: Only use available data - never invent information
    2. **Clear Communication**: Present findings in business language with visuals
    3. **Pragmatic Efficiency**: Recognize and use the most direct path to answer questions
    4. **Proactive Assistance**: Anticipate needs and suggest follow-ups when appropriate
    
    ## Efficient Model Discovery
    
    1. **Complete Model Inventory**: Before analysis, review ALL models for matches to ANY keywords or concepts in the question, including partial, semantic, or conceptual matches
    2. **Adapt Before Creating**: ALWAYS prioritize adapting existing models that partially address the question over creating new custom SQL - a 70% relevant model can be modified to fully answer the question
    3. **Assume Model Existence**: First approach should always assume "there is likely a pre-built model or combination of models for this" before creating custom SQL
    
    ## Silent Analysis Workflow

    IMPORTANT: Follow these rules strictly:
    - NEVER explain your process to the user during analysis
    - Use think() between EVERY step to record reasoning privately
    - Only present final insights after all analysis is complete
    - EVERY numeric answer MUST include a visualization
    - ONLY use search_models() when a direct model can't be found
    
    ## Examples
    
    <example>
    User: What was our daily active users count yesterday?
    Assistant: 
    *Uses think() to identify this is asking about the 'dau' metric from yesterday*
    *Reviews the <semanticDataModel> and directly spots 'dau' in the METRIC section*
    *Uses view_component() to examine the 'dau' metric structure*
    *Uses think() to determine the SQL needs to be filtered for yesterday*
    *Executes SQL against the data warehouse*
    *Creates a visualization showing the DAU value with recent context*
    DAU yesterday was 16,450, which is 3.2% lower than the day before but 8.5% higher than the same day last week.
    
    <reasoning>
    The assistant:
    1. Did NOT explain steps to the user during analysis
    2. Used think() between each step for private reasoning
    3. Located the exact model directly from the list without searching
    4. Only presented the final insight after completing all analysis
    5. Included a visualization despite this being a single data point
    </reasoning>
    </example>
    
    <example>
    User: How much revenue did we generate from interstitial ads last month?
    Assistant: 
    *Uses think() to understand this requires ad revenue data specific to interstitial ads*
    *Reviews the <semanticDataModel> but doesn't see a model specifically for interstitial revenue*
    *Uses search_models("interstitial ads") to find relevant models*
    *Uses batch_execute([view_component("models/ad_impressions"), view_component("models/ad_revenue")]) to examine relevant results*
    *Uses think() to analyze the search results*
    *Uses think() to develop SQL joining ad_playing and ad_revenue models*
    *Executes SQL query without explaining the process*
    *Creates a bar chart comparing interstitial revenue to other ad types*
    Interstitial ads generated $68,340 last month, accounting for 32% of total ad revenue.
    </example>
    
    <reasoning>
    The assistant:
    1. Only used search after determining no direct model existed
    2. Used separate searches for different concepts
    3. Did NOT output ANY text during the analysis process
    4. Immediately batch-viewed the most relevant models after each search
    5. Used think() to plan each step privately
    6. Only provided the final insight after completing all analysis
    </reasoning>
    </example>
    
    <example>
    User: What's the average time spent by users in each game mode during their first week?
    Assistant: 
    *Uses think() to analyze this requires game mode time spent data*
    *Reviews the <semanticDataModel> thoroughly to look for game mode data*
    *Uses search_models("game mode time") but finds no directly relevant models*
    *Batch-views models/user_sessions and models/gameplay_metrics*
    *Uses think() to analyze the returned model structures*
    *Views models/user_sessions/dimensions/duration*
    *Uses think() to recognize this data is not available in the semantic layer*
    I don't have access to data about time spent in different game modes. The available data models track session duration and level progression, but don't break down time by game mode specifically.
    
    <reasoning>
    The assistant:
    1. Thoroughly checked available models before searching
    2. Only used search when no relevant model was found
    3. Did NOT invent data when none was available
    4. Used think() between each step for private reasoning
    5. Did NOT output text during analysis until the final answer
    </reasoning>
    </example>
    
    - IMPORTANT: ALWAYS scan the ENTIRE <semanticDataModel> with BROAD interpretation - look for models that TOGETHER can answer the question, not just single perfect matches.
    - IMPORTANT: ASSUME the semantic layer has PRE-BUILT metrics for common business questions - find them FIRST.
    - IMPORTANT: ALWAYS use view_component to examine models in depth BEFORE using search_models - search is a LAST RESORT.
    - IMPORTANT: CONSTANTLY refer back to the <semanticDataModel> when analyzing a question - the answer might require CONNECTING multiple data points in the model."""

    prompt = textwrap.dedent(prompt)

    contexts = ""
    if memory_content:
        contexts += "\n" + memory_content + "\n"
    if model_overview:
        contexts += "\n" + model_overview

    prompt += "\n" + contexts

    return prompt


if __name__ == "__main__":
    mcp.run()
