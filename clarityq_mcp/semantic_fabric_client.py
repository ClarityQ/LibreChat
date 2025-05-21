import time
import traceback
from datetime import datetime
from functools import wraps
from typing import Callable, List, Optional, Tuple, Dict, TypeVar, Awaitable
from uuid import UUID

import httpx
from loguru import logger
from pydantic import TypeAdapter, BaseModel

from clarityq_mcp.semantic_fabric_config import API_URL, AUTH0_CLIENT_ID, AUTH0_CLIENT_SECRET, AUTH0_DOMAIN
from clarityq_mcp.semantic_fabric_models import (
    ComponentTypes,
    SemanticComponentUpdate,
    SemanticFabricSearchQuery,
    ComponentOrDimension,
    SemanticFabricResponse,
)


# Memory models
class MemoryCreate(BaseModel):
    content: str


class MemoryRead(BaseModel):
    id: str
    content: str
    product_id: UUID
    created_at: datetime
    last_updated: datetime


# Type variable for the return type of the function being decorated
T = TypeVar("T")


def handle_client_errors(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    """Decorator to handle HTTP and network errors consistently across client methods.

    This maintains the original return type while properly handling and logging errors.
    """

    @wraps(func)
    async def wrapper(*args, **kwargs) -> T:
        try:
            return await func(*args, **kwargs)
        except httpx.ReadTimeout as e:
            logger.error(f"ReadTimeout in {func.__name__}: {str(e)}")
            raise
        except httpx.ConnectTimeout as e:
            logger.error(f"ConnectTimeout in {func.__name__}: {str(e)}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Network error in {func.__name__}: {str(e)}")
            raise
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            try:
                error_json = e.response.json() if e.response.content else {}
                detail = error_json.get("detail", str(e))
            except Exception:
                detail = str(e)

            logger.error(f"HTTP error {status_code} in {func.__name__}: {detail}")
            raise
        except Exception as e:
            tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}\n{tb_str}")
            raise

    return wrapper


def format_http_error(error: httpx.HTTPStatusError) -> str:
    """Formats an HTTP error into a user-friendly message."""
    status_code = error.response.status_code
    try:
        error_json = error.response.json() if error.response.content else {}
        detail = error_json.get("detail", str(error))
    except Exception:
        detail = str(error)

    if 400 <= status_code < 500:
        if status_code == 400:
            return f"Bad Request (400): Invalid request. {detail}"
        elif status_code == 401:
            return f"Unauthorized (401): Authentication required. {detail}"
        elif status_code == 403:
            return f"Forbidden (403): Insufficient permissions. {detail}"
        elif status_code == 404:
            return f"Not Found (404): Resource not found. {detail}"
        else:
            return f"Client Error ({status_code}): {detail}"
    elif 500 <= status_code < 600:
        return f"Server Error ({status_code}): {detail}"
    else:
        return f"HTTP Error ({status_code}): {detail}"


def format_client_error(error: Exception, context: str = "") -> str:
    """Formats any client error into a user-friendly message.

    Args:
        error: The exception to format
        context: Optional context string to customize error messages (e.g., "executing SQL")
    """
    context_str = f" while {context}" if context else ""

    if isinstance(error, httpx.ReadTimeout):
        return f"Error: Connection to API timed out{context_str}. Please try again later."
    elif isinstance(error, httpx.ConnectTimeout):
        return f"Error: Could not connect to API (connection timeout){context_str}. Please try again later."
    elif isinstance(error, httpx.RequestError):
        return f"Network Error: Could not connect to the API{context_str}. {str(error)}"
    elif isinstance(error, httpx.HTTPStatusError):
        error_msg = format_http_error(error)
        # Customize specific error messages based on context
        if context == "executing SQL" and error.response.status_code == 400:
            error_msg = error_msg.replace("Bad Request (400): Invalid request", "Bad Request (400): Invalid SQL query")
        return error_msg
    else:
        return f"Error{context_str}: {str(error)}"


class SemanticFabricClient:
    __slots__ = ["base_url", "client", "access_token", "auth_credentials", "token_expiry"]

    def __init__(
        self,
        base_url: str = None,
        auth0_client_id: str = None,
        auth_client_secret: str = None,
        auth0_domain: str = None,
    ):
        # Check base_url from params or environment
        self.base_url = base_url or API_URL
        if not self.base_url:
            raise ValueError("base_url must be provided either directly or via API_URL config")

        self.client = httpx.AsyncClient(base_url=self.base_url)
        auth0_client_id = auth0_client_id or AUTH0_CLIENT_ID
        auth_client_secret = auth_client_secret or AUTH0_CLIENT_SECRET
        auth0_domain = auth0_domain or AUTH0_DOMAIN

        # Initialize access token if all auth params are available
        self.access_token = None
        self.token_expiry = 0
        self.auth_credentials = None

        if auth0_client_id and auth_client_secret and auth0_domain:
            self.auth_credentials = {
                "client_id": auth0_client_id,
                "client_secret": auth_client_secret,
                "domain": auth0_domain,
            }
            self._refresh_access_token()
        # If any auth parameter is provided but not all, raise an error
        elif any([auth0_client_id, auth_client_secret, auth0_domain]):
            raise ValueError(
                "All Auth0 credentials (auth0_client_id, auth_client_secret, auth0_domain) must be provided "
                "either directly or via environment variables for authentication"
            )

    def _refresh_access_token(self) -> None:
        if not self.auth_credentials:
            logger.warning("Cannot refresh token: auth credentials not available")
            return

        token_data = self._get_access_token(
            self.auth_credentials["client_id"], self.auth_credentials["client_secret"], self.auth_credentials["domain"]
        )
        self.access_token = token_data["access_token"]
        self.token_expiry = time.time() + token_data["expires_in"] - 30
        logger.debug(f"Token refreshed, expires at: {datetime.fromtimestamp(self.token_expiry).isoformat()}")

    @staticmethod
    def _get_access_token(auth0_client_id: str, auth_client_secret: str, auth0_domain: str) -> Dict:
        audience = f"{auth0_domain}/api/v2/"
        token_url = f"{auth0_domain}/oauth/token"
        headers = {"content-type": "application/json"}
        payload = {
            "client_id": auth0_client_id,
            "client_secret": auth_client_secret,
            "audience": audience,
            "grant_type": "client_credentials",
        }
        response = httpx.post(token_url, json=payload, headers=headers)
        assert response.status_code == 200
        return response.json()

    async def _ensure_valid_token(self) -> None:
        if not self.auth_credentials:
            return

        if not self.access_token or time.time() >= self.token_expiry:
            logger.info("Access token expired or not set, refreshing")
            self._refresh_access_token()

    async def request_with_auth(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Make an HTTP request with auth token, handling token refresh if needed."""
        await self._ensure_valid_token()

        if self.access_token and "headers" not in kwargs:
            kwargs["headers"] = {}

        if self.access_token and kwargs.get("headers") is not None:
            kwargs["headers"]["Authorization"] = f"Bearer {self.access_token}"

        try:
            response = await getattr(self.client, method)(url=url, **kwargs)
            return response
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401 and self.auth_credentials:
                logger.warning("Received 401 Unauthorized, refreshing token and retrying")
                self._refresh_access_token()

                # Update headers with new token
                if "headers" not in kwargs:
                    kwargs["headers"] = {}
                kwargs["headers"]["Authorization"] = f"Bearer {self.access_token}"

                return await getattr(self.client, method)(url=url, **kwargs)
            raise

    @handle_client_errors
    async def create_component(self, product_id: UUID, component: ComponentTypes) -> ComponentTypes:
        """Create a new semantic component.

        Args:
            product_id: The UUID of the product to create the component in
            component: The component definition

        Returns:
            The created component

        Raises:
            ValidationError: If the component can't be validated
            httpx.HTTPStatusError: For HTTP errors
            httpx.ReadTimeout: If the request times out
            httpx.ConnectTimeout: If connection times out
            httpx.RequestError: For other network-related errors
        """
        logger.info(f"Creating component - product_id: {product_id}, component: {component}")

        response = await self.request_with_auth(
            "post",
            url=f"/products/{product_id}/semantic-fabric",
            json=component.model_dump(),
        )
        response.raise_for_status()
        response_json = response.json()
        component_adapter = TypeAdapter(ComponentTypes)
        result = component_adapter.validate_python(response_json)

        logger.info(f"Component created successfully: {result}")
        return result

    @handle_client_errors
    async def update_component_by_name(
        self,
        product_id: UUID,
        component_name: str,
        dimension_name: Optional[str] = None,
        recipe_name: Optional[str] = None,
        component: SemanticComponentUpdate = None,
    ) -> Optional[ComponentOrDimension]:
        """Update a component, dimension, or recipe by name.

        Args:
            product_id: The UUID of the product
            component_name: The name of the component to update
            dimension_name: Optional name of dimension to update
            recipe_name: Optional name of recipe to update
            component: The update specification

        Returns:
            The updated component, dimension, or recipe

        Raises:
            ValidationError: If the response can't be validated
            httpx.HTTPStatusError: For HTTP errors
            httpx.ReadTimeout: If the request times out
            httpx.ConnectTimeout: If connection times out
            httpx.RequestError: For other network-related errors
        """
        logger.info(
            f"Updating component by name - product_id: {product_id}, component_name: {component_name}, "
            f"dimension_name: {dimension_name}, recipe_name: {recipe_name}, update: {component}"
        )

        params = {}
        if dimension_name:
            params["dimension_name"] = dimension_name
        if recipe_name:
            params["recipe_name"] = recipe_name

        response = await self.request_with_auth(
            "put",
            url=f"/products/{product_id}/semantic-fabric/by-name/{component_name}",
            json=component.model_dump(exclude_unset=True, exclude_none=True),
            params=params,
        )

        response.raise_for_status()
        response_json = response.json()
        result = SemanticFabricResponse.model_validate(response_json)
        logger.info(f"Component updated by name successfully: {result}")
        return result

    @handle_client_errors
    async def delete_component_by_name(
        self,
        product_id: UUID,
        component_name: str,
        dimension_name: Optional[str] = None,
        recipe_name: Optional[str] = None,
    ) -> bool:
        """Delete a component, dimension, or recipe by name.

        Args:
            product_id: The UUID of the product
            component_name: The name of the component to delete
            dimension_name: Optional name of dimension to delete
            recipe_name: Optional name of recipe to delete

        Returns:
            True if deletion was successful, False if component/dimension/recipe not found

        Raises:
            httpx.HTTPStatusError: For HTTP errors (except 404)
            httpx.ReadTimeout: If the request times out
            httpx.ConnectTimeout: If connection times out
            httpx.RequestError: For other network-related errors
        """
        logger.info(
            f"Deleting component by name - product_id: {product_id}, component_name: {component_name}, "
            f"dimension_name: {dimension_name}, recipe_name: {recipe_name}"
        )

        params = {}
        if dimension_name:
            params["dimension_name"] = dimension_name
        if recipe_name:
            params["recipe_name"] = recipe_name

        response = await self.request_with_auth(
            "delete",
            url=f"/products/{product_id}/semantic-fabric/by-name/{component_name}",
            params=params,
        )
        if response.status_code == 404:
            logger.info(f"Component not found for deletion by name: {component_name}")
            return False

        response.raise_for_status()
        result = response.status_code == 200

        logger.info(f"Component deletion by name result: {result}")
        return result

    @handle_client_errors
    async def search_components(
        self, product_id: UUID, query: SemanticFabricSearchQuery
    ) -> Tuple[List[ComponentTypes], int]:
        """Search for components based on query criteria.

        Args:
            product_id: The UUID of the product
            query: Search query parameters

        Returns:
            Tuple of matching components list and total count

        Raises:
            ValidationError: If components can't be validated
            httpx.HTTPStatusError: For HTTP errors
            httpx.ReadTimeout: If the request times out
            httpx.ConnectTimeout: If connection times out
            httpx.RequestError: For other network-related errors
        """
        logger.info(f"Searching components - product_id: {product_id}, query: {query}")
        response = await self.request_with_auth(
            "post",
            url=f"/products/{product_id}/semantic-fabric/search",
            json=query.model_dump(),
        )
        response.raise_for_status()
        response_json = response.json()
        components_data = response_json["data"]
        components_adapter = TypeAdapter(List[ComponentTypes])
        components = components_adapter.validate_python(components_data)
        total_count = response_json["total_count"]
        logger.info(f"Search completed - found {total_count} components")
        return components, total_count

    @handle_client_errors
    async def get_component_by_name(self, product_id: UUID, name: str) -> Optional[ComponentTypes]:
        """Get a component by its name.

        Args:
            product_id: The UUID of the product
            name: The name of the component to retrieve

        Returns:
            The component if found, None otherwise

        Raises:
            ValidationError: If the component can't be validated
            httpx.HTTPStatusError: For HTTP errors
            httpx.ReadTimeout: If the request times out
            httpx.ConnectTimeout: If connection times out
            httpx.RequestError: For other network-related errors
        """
        logger.info(f"Getting component by name - product_id: {product_id}, name: {name}")

        response = await self.request_with_auth(
            "post",
            url=f"/products/{product_id}/semantic-fabric/by-name",
            json={"name": name},
        )
        response.raise_for_status()
        response_json = response.json()
        component_adapter = TypeAdapter(ComponentTypes)
        result = component_adapter.validate_python(response_json)

        logger.info(f"Component retrieved by name successfully: {result}")
        return result

    @handle_client_errors
    async def get_all_components(self, product_id: UUID) -> List[ComponentTypes]:
        """Gets all components for a product.

        Args:
            product_id: The UUID of the product

        Returns:
            List of all components in the product

        Raises:
            ValidationError: If any component in the response cannot be validated
            httpx.HTTPStatusError: For HTTP errors
            httpx.ReadTimeout: If the request times out
            httpx.ConnectTimeout: If connection times out
            httpx.RequestError: For other network-related errors
        """

        logger.info(f"Getting all components - product_id: {product_id}")
        response = await self.request_with_auth("get", url=f"/products/{product_id}/semantic-fabric/bulk")
        response.raise_for_status()
        response_json = response.json()
        components_adapter = TypeAdapter(List[ComponentTypes])
        result = components_adapter.validate_python(response_json)
        logger.info(f"Retrieved {len(result)} total components")
        return result

    @handle_client_errors
    async def set_component(
        self,
        product_id: UUID,
        component_id: str,
        component: SemanticComponentUpdate,
    ) -> Optional[ComponentOrDimension]:
        raise NotImplementedError

    @handle_client_errors
    async def set_component_by_name(
        self,
        product_id: UUID,
        component_name: str,
        dimension_name: Optional[str] = None,
        recipe_name: Optional[str] = None,
        component: SemanticComponentUpdate = None,
    ) -> Optional[ComponentOrDimension]:
        raise NotImplementedError

    @handle_client_errors
    async def execute_sql_query(self, query: str, product_id: str) -> Dict:
        """Execute a SQL query against the data warehouse (READ ONLY).

        Args:
            query: The SQL query to execute
            product_id: The product ID to execute the query against

        Returns:
            Dict containing the query results

        Raises:
            httpx.HTTPStatusError: For HTTP errors
            httpx.ReadTimeout: If the request times out
            httpx.ConnectTimeout: If connection times out
            httpx.RequestError: For other network-related errors
        """
        logger.info(f"Executing SQL query - product_id: {product_id}")
        payload = {"query": query, "product_id": product_id, "validate_on_catalog": False}

        # Use a 5-minute (300 seconds) timeout for SQL queries
        response = await self.request_with_auth(
            "post",
            url="runnable/workflow/fetch-data-from-data-warehouse",
            json=payload,
            timeout=300.0,
        )
        response.raise_for_status()
        response_json = response.json()
        logger.info("SQL query executed successfully")

        # Return the output if it exists, otherwise return the whole response
        if "output" in response_json:
            return response_json["output"]
        return response_json

    @handle_client_errors
    async def set_memory(self, product_id: UUID, memory: MemoryCreate) -> MemoryRead:
        """Set (create or update) the memory for a product.

        Args:
            product_id: The UUID of the product
            memory: The memory content to set

        Returns:
            The created or updated memory

        Raises:
            httpx.HTTPStatusError: For HTTP errors
            httpx.ReadTimeout: If the request times out
            httpx.ConnectTimeout: If connection times out
            httpx.RequestError: For other network-related errors
        """
        logger.info(f"Setting memory - product_id: {product_id}, content: {memory.content[:50]}...")

        response = await self.request_with_auth(
            "put",
            url=f"/products/{product_id}/memory",
            json=memory.model_dump(),
        )
        response.raise_for_status()
        response_json = response.json()
        result = MemoryRead.model_validate(response_json)

        logger.info(f"Memory set successfully for product {product_id}")
        return result

    @handle_client_errors
    async def get_memory(self, product_id: UUID) -> Optional[MemoryRead]:
        """Get the memory for a product.

        Args:
            product_id: The UUID of the product

        Returns:
            The memory if found, None otherwise

        Raises:
            httpx.HTTPStatusError: For HTTP errors (except 404)
            httpx.ReadTimeout: If the request times out
            httpx.ConnectTimeout: If connection times out
            httpx.RequestError: For other network-related errors
        """
        logger.info(f"Getting memory - product_id: {product_id}")

        try:
            response = await self.request_with_auth(
                "get",
                url=f"/products/{product_id}/memory",
            )
            response.raise_for_status()
            response_json = response.json()
            result = MemoryRead.model_validate(response_json)

            logger.info(f"Memory retrieved successfully for product {product_id}")
            return result
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.info(f"No memory found for product {product_id}")
                return None
            raise

    @handle_client_errors
    async def clear_memory(self, product_id: UUID) -> bool:
        """Clear the memory for a product.

        Args:
            product_id: The UUID of the product

        Returns:
            True if the memory was cleared, False if no memory existed

        Raises:
            httpx.HTTPStatusError: For HTTP errors
            httpx.ReadTimeout: If the request times out
            httpx.ConnectTimeout: If connection times out
            httpx.RequestError: For other network-related errors
        """
        logger.info(f"Clearing memory - product_id: {product_id}")

        response = await self.request_with_auth(
            "delete",
            url=f"/products/{product_id}/memory",
        )
        response.raise_for_status()
        result = response.json()

        logger.info(f"Memory cleared successfully for product {product_id}")
        return result
