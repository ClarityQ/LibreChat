from typing import List, Any, Dict, Optional
from uuid import UUID

import yaml
import textwrap
from loguru import logger

from clarityq_mcp.semantic_fabric_client import SemanticFabricClient, MemoryRead
from clarityq_mcp.semantic_fabric_config import PRODUCT_ID
from clarityq_mcp.semantic_fabric_models import (
    SemanticFabricResponse,
    NodeType,
    UpdateOperation,
    PropertiesUpdateTypes,
    ComponentOrDimension,
    ComponentTypes,
)


def render_memory(memory: MemoryRead) -> str:
    """Renders memory content with a descriptive header into a context block.

    Args:
        memory: The memory object containing content to render

    Returns:
        Formatted memory context block with description
    """
    if not memory:
        return ""

    explanation = (
        "IMPORTANT: This memory context contains persistent information that OVERRIDES default behavior. "
        "It includes user preferences, domain-specific notes, and configuration settings that should be "
        "prioritized over other contexts. Always follow these instructions exactly as written."
    )

    return f'{explanation}\n\n<context name="memory">\n{memory.content}\n</context>'


def render_data_model_overview(components: List[SemanticFabricResponse], include_descriptions: bool = True) -> str:
    """Renders a list of Semantic Fabric components into a formatted context block.

    Args:
        components: List of Semantic Fabric component responses
        include_descriptions: Whether to include descriptions for dimensions and recipes (default: True)

    Returns:
        Formatted semantic data model context block
    """
    components_by_type = {}
    for component in components:
        component_type = component.semantic_type
        if component_type not in components_by_type:
            components_by_type[component_type] = []

        components_by_type[component_type].append(component)

    ordered_types = [
        NodeType.ENTITY,
        NodeType.HIERARCHY,
        NodeType.FEATURE,
        NodeType.DIMENSION,
        NodeType.METRIC,
        NodeType.SEGMENT,
        NodeType.FUNNEL,
        NodeType.TIME,
        NodeType.RECIPE,
    ]

    # Filter to include only types that have components
    active_types = [t for t in ordered_types if t in components_by_type and components_by_type[t]]

    # Build structured data for YAML
    yaml_data = {}

    for node_type in active_types:
        type_components = components_by_type.get(node_type, [])
        if not type_components:
            continue

        # Create an entry for this type - convert NodeType to string
        type_name = str(node_type)
        yaml_data[type_name] = {}

        # Add each component under this type
        sorted_components = sorted(type_components, key=lambda c: c.properties.name)
        for component in sorted_components:
            component_dict = component.model_dump(
                mode="json", exclude={"id", "created_at", "last_updated", "product_id"}
            )
            properties = component_dict.get("properties", {}) or {}
            component_name = properties.get("name", "")
            description = properties.get("description", "")

            # Use just the component name as the key
            yaml_data[type_name][component_name] = {}

            # Add description as a property if it exists
            if description:
                yaml_data[type_name][component_name]["description"] = description

            # Add dimensions if they exist
            dimensions = properties.get("dimensions") or {}
            recipes = properties.get("recipes") or {}

            component_details = {}

            if dimensions:
                if include_descriptions:
                    # Include dimension names with descriptions
                    dimension_dict = {}
                    for dim_name, dim_data in dimensions.items():
                        # Extract dimension description if available
                        dim_description = ""
                        if isinstance(dim_data, dict):
                            if "properties" in dim_data and isinstance(dim_data["properties"], dict):
                                dim_description = dim_data["properties"].get("description", "")
                            else:
                                dim_description = dim_data.get("description", "")

                        dimension_dict[dim_name] = dim_description if dim_description else None

                    if dimension_dict:
                        component_details["dimensions"] = dimension_dict
                else:
                    # Just include dimension names as a list
                    dimension_names = list(dimensions.keys())
                    if dimension_names:
                        component_details["dimensions"] = dimension_names

            # Add recipes if they exist
            if recipes:
                if include_descriptions:
                    # Include recipe names with descriptions
                    recipe_dict = {}
                    for recipe_name, recipe_data in recipes.items():
                        # Extract recipe description if available
                        recipe_description = ""
                        if isinstance(recipe_data, dict):
                            if "properties" in recipe_data and isinstance(recipe_data["properties"], dict):
                                recipe_description = recipe_data["properties"].get("description", "")
                            else:
                                recipe_description = recipe_data.get("description", "")

                        recipe_dict[recipe_name] = recipe_description if recipe_description else None

                    if recipe_dict:
                        component_details["recipes"] = recipe_dict
                else:
                    # Just include recipe names as a list
                    recipe_names = list(recipes.keys())
                    if recipe_names:
                        component_details["recipes"] = recipe_names

            if component_details:
                yaml_data[type_name][component_name].update(component_details)

    # Convert to YAML string
    yaml_output = yaml.dump(
        yaml_data,
        default_flow_style=False,
        sort_keys=False,
        width=80,
        indent=2,
    )

    description_suffix = " with descriptions" if include_descriptions else ""
    explanation = (
        "Structured overview of Semantic Fabric data model. YAML shows components by type with names, "
        f"descriptions, dimensions{description_suffix}, and recipes{description_suffix}.\n\n"
        "IMPORTANT:\n"
        '- Names are display titles; check "sql" field for actual implementation\n'
        '- View model details: view_component("models/<model_name>")\n'
        '- View dimension: view_component("models/<model_name>/dimensions/<dimension_name>")\n'
        '- View recipe: view_component("models/<model_name>/recipes/<recipe_name>")'
    )

    return f'<context name="semanticDataModel">\n{explanation}\n\n```yaml\n{yaml_output}```\n</context>'


def deep_merge(source: Any, target: Any) -> Any:
    if isinstance(source, dict) and isinstance(target, dict):
        # Create a new dict to avoid modifying the original
        result = target.copy()
        # Process all keys from source
        for key, value in source.items():
            # Three conditions for recursive merge:
            # 1. Key exists in both dictionaries
            # 2. Values are the same type (both dicts or both lists)
            # 3. The type is something we can merge (dict or list)
            if key in result and isinstance(result[key], type(value)) and isinstance(value, (dict, list)):
                # Recursively merge the nested structures
                result[key] = deep_merge(value, result[key])
            else:
                # Otherwise use the source value (overwrite or add new)
                result[key] = value
        return result
    elif isinstance(source, list) and isinstance(target, list):
        # For lists, we concatenate them (keeping target items first)
        return target + source
    else:
        # For incompatible types or primitives, the source value wins
        return source


def deep_remove(target: Any, to_remove: Any) -> Any:
    """
    Deep remove values from a target object.

    Args:
        target: The target object to remove from
        to_remove: Specification of what to remove

    Returns:
        Modified object with values removed
    """
    if to_remove is None:
        # None means remove everything
        return None

    if isinstance(to_remove, list) and isinstance(target, dict):
        # Remove specified keys from dict
        result = target.copy()
        for key in to_remove:
            if key in result:
                del result[key]
        return result

    elif isinstance(to_remove, list) and isinstance(target, list):
        # Remove specified items from list
        result = target.copy()
        for item in to_remove:
            if item in result:
                result.remove(item)
        return result

    elif isinstance(to_remove, dict) and isinstance(target, dict):
        # Process nested removals for each key in the removal dict
        result = target.copy()
        for key, value in to_remove.items():
            if key in result:
                if value is None:
                    # Remove entire key
                    del result[key]
                elif isinstance(value, (dict, list)) and isinstance(result[key], type(value)):
                    # Recursive removal
                    modified = deep_remove(result[key], value)
                    if modified is None or (isinstance(modified, (dict, list)) and len(modified) == 0):
                        del result[key]
                    else:
                        result[key] = modified

        return result

    else:
        # For other cases, return the target unchanged
        return target


def apply_patch_by_operation_type(
    operation: UpdateOperation,
    current_model: ComponentOrDimension,
    update_properties: PropertiesUpdateTypes,
) -> Dict[str, Any]:
    component_dict = current_model.model_dump()
    properties_update = update_properties.model_dump(
        mode="json",
        exclude_unset=True,
        exclude_none=True,
        exclude_defaults=True,
        exclude={"semantic_type"},
    )
    properties = component_dict.setdefault("properties", {})

    logger.info(f"Source properties: {properties}")
    logger.info(f"Update properties: {properties_update}")

    if operation == UpdateOperation.SET:
        # SET operation: Directly overwrite values with new ones
        # Any existing keys will be replaced, new keys will be added
        for key, value in properties_update.items():
            properties[key] = value

    elif operation == UpdateOperation.ADD:
        # ADD operation: Intelligently merge values based on their types
        for key, value in properties_update.items():
            # Case 1: Key doesn't exist or is None - simply assign the new value
            if key not in properties or properties[key] is None:
                properties[key] = value
                continue

            # Case 2: Key exists and isn't None - deep merge the values
            # value takes precedence over existing properties[key]
            properties[key] = deep_merge(value, properties[key])

    elif operation == UpdateOperation.REMOVE:
        # REMOVE operation: Remove specified values or entire keys
        for key, value in properties_update.items():
            if key in properties:
                # Case 1: If value is None, remove the entire key
                if value is None:
                    del properties[key]
                # Case 2: Perform deep removal based on specified value
                else:
                    result = deep_remove(properties[key], value)

                    # If result is empty or None after removal, delete the entire key
                    if result is None or (isinstance(result, (dict, list)) and len(result) == 0):
                        del properties[key]
                    else:
                        # Otherwise, update with the modified value
                        properties[key] = result

    # Create and log a model instance with the modified dictionary
    logger.debug(f"Updated properties: {properties}")
    # Return the properties dictionary directly
    return properties


def render_search_results(components: List[SemanticFabricResponse], search_text: Optional[str] = None) -> str:
    """Renders search results in a terminal-friendly numbered list format.

    Args:
        components: List of Semantic Fabric component responses from search
        search_text: Optional search text used for the query (will be included in header)

    Returns:
        Formatted terminal-friendly search results with components numbered
    """
    if not components:
        if search_text:
            empty_result = f"No models found matching search: '{search_text}'"
            return f'<search-results query="{search_text or ""}" count="0">\n{empty_result}\n</search-results>'
        return '<search-results query="" count="0">\nNo models found\n</search-results>'

    # Create header
    search_header = f"Search results for '{search_text}'" if search_text else "Search results"
    result = [f"# {search_header} ({len(components)} matches)"]
    result.append("")

    # Use a counter to number results sequentially
    counter = 1

    # Process each component in the original order they came from the search
    for component in components:
        # Extract component type for display
        component_type = component.semantic_type
        name = component.properties.name
        description = component.properties.description or ""

        # Get dimensions and recipes if they exist
        dimensions = getattr(component.properties, "dimensions", None) or {}
        recipes = getattr(component.properties, "recipes", None) or {}

        # Truncate description if too long, and wrap it for better display
        if description:
            description = description.replace("\n", " ")
            if len(description) > 120:
                description = description[:117] + "..."
            description = textwrap.fill(description, width=80, initial_indent="   ", subsequent_indent="   ")

        # Build the component entry with bold name using markdown
        comp_entry = [f"{counter}. **{name}** ({component_type})"]
        if description:
            comp_entry.append(description)

        # Add dimensions if they exist
        if dimensions:
            dim_count = len(dimensions)
            dim_list = list(dimensions.keys())
            dim_preview = ", ".join([f"`{dim}`" for dim in dim_list[:3]])
            if dim_count > 3:
                dim_preview += f", ... (+{dim_count - 3} more)"
            comp_entry.append(f"   **Dimensions**: {dim_preview}")

        # Add recipes if they exist
        if recipes:
            recipe_count = len(recipes)
            recipe_list = list(recipes.keys())
            recipe_preview = ", ".join([f"`{recipe}`" for recipe in recipe_list[:3]])
            if recipe_count > 3:
                recipe_preview += f", ... (+{recipe_count - 3} more)"
            comp_entry.append(f"   **Recipes**: {recipe_preview}")

        # Add component entry to results
        result.append("\n".join(comp_entry))
        result.append("")

        # Increment the counter
        counter += 1

    # Add usage hints
    usage_notice = '\n<clarityq_automated_notice visibility="system_only">\n'
    usage_notice += "For efficiency, batch-view most relevant (3-5) models together: `batch_execute([view_component(\"models/...\")])`\n"
    usage_notice += "</clarityq_automated_notice>\n"
    formatted_result = "\n".join(result)

    # Wrap the entire content in search-results XML tags
    return (
        f'<search-results query="{search_text or ""}" count="{len(components)}">\n{formatted_result}\n</search-results>'
    )


def render_component(component: ComponentTypes) -> str:
    component_dict = component.model_dump(mode="json", exclude={"id", "created_at", "last_updated", "product_id"})
    properties = component_dict.get("properties") or {}

    # Handle dimensions
    dimensions = properties.get("dimensions") or {}
    if dimensions:
        simplified_dimensions = {}
        for dimension_name, dimension_data in dimensions.items():
            if isinstance(dimension_data, dict) and "properties" in dimension_data:
                simplified_dimensions[dimension_name] = dimension_data["properties"].get("description", "")
            elif isinstance(dimension_data, dict):
                simplified_dimensions[dimension_name] = dimension_data.get("description", "")
            else:
                simplified_dimensions[dimension_name] = ""
        component_dict["properties"]["dimensions"] = simplified_dimensions

    # Handle recipes
    recipes = properties.get("recipes") or {}
    if recipes:
        simplified_recipes = {}
        for recipe_name, recipe_data in recipes.items():
            if isinstance(recipe_data, dict) and "properties" in recipe_data:
                simplified_recipes[recipe_name] = recipe_data["properties"].get("description", "")
            elif isinstance(recipe_data, dict):
                simplified_recipes[recipe_name] = recipe_data.get("description", "")
            else:
                simplified_recipes[recipe_name] = ""
        component_dict["properties"]["recipes"] = simplified_recipes
    return yaml.dump(
        component_dict,
        default_flow_style=False,  # Use block style for better readability
        sort_keys=False,  # Preserve the order of keys we defined
        width=80,  # Reasonable line width
        indent=2,  # Standard indentation
    )


async def view_model(model_name: str) -> str:
    """View a complete semantic model definition.

    Returns the full details of the specified model, including all its properties,
    dimensions (for entities and features), and other attributes.

    Args:
        model_name: Name of the model to view

    Returns:
        Full detailed information about the requested model in a human-readable format
    """
    client = SemanticFabricClient()
    component = await client.get_component_by_name(product_id=UUID(PRODUCT_ID), name=model_name)
    if not component:
        return f"Model not found with name: {model_name}"

    result = render_component(component)

    # Add reminder if the component has dimensions
    dimensions = getattr(component.properties, "dimensions", None) or {}
    recipes = getattr(component.properties, "recipes", None) or {}

    if dimensions or recipes:
        reminder = '\n\n<clarityq_system_notice visibility="system_only">\n'
        # Add dimension info if present
        if dimensions:
            reminder += (
                f"• DIMENSIONS [{len(dimensions)}]:\n"
                f"  - NOT SQL columns! These are semantic references requiring individual exploration\n"
                f'  - For dimension details: view_component("models/{model_name}/dimensions/[name]")\n'
            )
        # Add recipe info if present
        if recipes:
            reminder += (
                f"• RECIPES [{len(recipes)}]:\n"
                f"  - Pre-defined semantic queries and transformations for this model\n"
                f'  - For recipe details: view_component("models/{model_name}/recipes/[name]")\n'
            )
        reminder += "• SQL ACCESS: Look for the 'sql' property to understand the underlying data structure\n"
        reminder += "</clarityq_system_notice>"

        result += reminder

    return result


async def view_dimension(model_name: str, dimension_name: str) -> str:
    """View a dimension within a model.

    Returns the full details of the specified dimension within the model.

    Args:
        model_name: Name of the model containing the dimension
        dimension_name: Name of the dimension to view

    Returns:
        Full detailed information about the requested dimension in a human-readable format
    """
    client = SemanticFabricClient()

    # First get the parent component
    component = await client.get_component_by_name(product_id=UUID(PRODUCT_ID), name=model_name)
    if not component:
        return f"Model not found with name: {model_name}"

    # Check if this component has the requested dimension
    dimensions = None
    if hasattr(component.properties, "dimensions"):
        dimensions = component.properties.dimensions

    if not dimensions or dimension_name not in dimensions:
        return f"No dimension named '{dimension_name}' found in model '{model_name}'"

    dimension = dimensions[dimension_name]
    result = render_component(dimension)

    return result


async def view_recipe(model_name: str, recipe_name: str) -> str:
    """View a recipe within a model.

    Returns the full details of the specified recipe within the model.

    Args:
        model_name: Name of the model containing the recipe
        recipe_name: Name of the recipe to view

    Returns:
        Full detailed information about the requested recipe in a human-readable format
    """
    client = SemanticFabricClient()

    # First get the parent component
    component = await client.get_component_by_name(product_id=UUID(PRODUCT_ID), name=model_name)
    if not component:
        return f"Model not found with name: {model_name}"

    # Check if this component has the requested recipe
    recipes = None
    if hasattr(component.properties, "recipes"):
        recipes = component.properties.recipes

    if not recipes or recipe_name not in recipes:
        return f"No recipe named '{recipe_name}' found in model '{model_name}'"

    recipe = recipes[recipe_name]
    result = render_component(recipe)

    return result


async def delete_model(model_name: str) -> str:
    """Delete a semantic model.

    This function deletes an entire model from the semantic layer.
    This operation cannot be undone.

    Args:
        model_name: Name of the model to delete

    Returns:
        Confirmation of the deletion
    """
    client = SemanticFabricClient()

    result = await client.delete_component_by_name(
        product_id=UUID(PRODUCT_ID), component_name=model_name, dimension_name=None
    )

    if not result:
        return f"Model not found with name: {model_name}"

    return f"Successfully deleted model '{model_name}'"


async def delete_dimension(model_name: str, dimension_name: str) -> str:
    """Delete a dimension from an entity or feature model.

    This function removes a dimension from an entity or feature model.
    This operation cannot be undone.

    Args:
        model_name: Name of the entity or feature model containing the dimension
        dimension_name: Name of the dimension to delete

    Returns:
        Confirmation of the deletion
    """
    client = SemanticFabricClient()

    result = await client.delete_component_by_name(
        product_id=UUID(PRODUCT_ID), component_name=model_name, dimension_name=dimension_name
    )

    if not result:
        return f"Model '{model_name}' or dimension '{dimension_name}' not found"

    return f"Successfully deleted dimension '{dimension_name}' from model '{model_name}'"


async def delete_recipe(model_name: str, recipe_name: str) -> str:
    """Delete a recipe from a model.

    This function removes a recipe from a model.
    This operation cannot be undone.

    Args:
        model_name: Name of the model containing the recipe
        recipe_name: Name of the recipe to delete

    Returns:
        Confirmation of the deletion
    """
    client = SemanticFabricClient()

    result = await client.delete_component_by_name(
        product_id=UUID(PRODUCT_ID), component_name=model_name, recipe_name=recipe_name
    )

    if not result:
        return f"Model '{model_name}' or recipe '{recipe_name}' not found"

    return f"Successfully deleted recipe '{recipe_name}' from model '{model_name}'"
