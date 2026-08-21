"""Storefront connectors: one port, three marketplaces.

Every channel exposes the same five verbs -- publish a listing, adjust
inventory, pull new orders, acknowledge an order, submit tracking. Above this
module nothing knows which marketplace it is talking to, so adding a fourth
channel is a new subclass and nothing else.

:class:`InMemoryChannel` is the reference adapter. It is not a mock: it
implements the full contract, holds real state, and is what the demo and the
tests run against. That is what lets the whole system be exercised end to end
with no accounts and no credentials.

:class:`ShopifyConnector` is the live one. It speaks the GraphQL Admin API over
``urllib`` from the standard library, so the package still has no third-party
dependencies. Amazon and eBay ship as ports with the same surface: their auth
flows (LWA and OAuth respectively) are substantial enough that stubbing them
convincingly would be worse than leaving an honest ``NotImplementedError``.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

from .domain import Channel, Listing, ListingState, Money, Order, Shipment
from .security import (
    IdempotencyGuard,
    RateLimiter,
    SecretRef,
    SecretResolver,
    idempotency_key,
)


class ChannelError(RuntimeError):
    """A marketplace rejected a request or could not be reached."""


class ChannelConnector(ABC):
    """The contract every storefront adapter implements."""

    channel: Channel

    @abstractmethod
    def publish_listing(self, listing: Listing) -> str:
        """Create or update the listing. Returns the marketplace's id."""

    @abstractmethod
    def update_inventory(self, sku: str, quantity: int) -> bool:
        """Set available quantity. Returns whether it took effect."""

    @abstractmethod
    def fetch_orders(self, since: datetime | None = None) -> list[Order]:
        """New orders placed after ``since``."""

    @abstractmethod
    def acknowledge_order(self, order: Order) -> bool:
        """Tell the marketplace we have accepted the order."""

    @abstractmethod
    def submit_tracking(self, order: Order, shipment: Shipment) -> bool:
        """Mark the order shipped and attach tracking."""

    def is_live(self) -> bool:
        """Whether this adapter reaches a real marketplace."""
        return False


# -- reference adapter -----------------------------------------------------


@dataclass
class InMemoryChannel(ChannelConnector):
    """Full-contract adapter backed by dictionaries.

    Records everything it was asked to do so tests and the demo can assert on
    real behaviour rather than on call counts.
    """

    channel: Channel = Channel.SHOPIFY
    published: dict[str, Listing] = field(default_factory=dict)
    inventory: dict[str, int] = field(default_factory=dict)
    inbox: list[Order] = field(default_factory=list)
    acknowledged: list[str] = field(default_factory=list)
    tracking: dict[str, Shipment] = field(default_factory=dict)
    _counter: int = field(default=0, init=False)

    def publish_listing(self, listing: Listing) -> str:
        if listing.state is ListingState.BLOCKED:
            raise ChannelError(f"refusing to publish blocked listing {listing.sku}")
        self._counter += 1
        external_id = f"{self.channel.value}-{self._counter:05d}"
        listing.external_id = external_id
        listing.state = ListingState.PUBLISHED
        self.published[listing.sku] = listing
        self.inventory[listing.sku] = listing.quantity
        return external_id

    def update_inventory(self, sku: str, quantity: int) -> bool:
        if sku not in self.published:
            return False
        self.inventory[sku] = max(0, quantity)
        self.published[sku].quantity = self.inventory[sku]
        return True

    def fetch_orders(self, since: datetime | None = None) -> list[Order]:
        ready = [o for o in self.inbox if since is None or o.placed_at > since]
        self.inbox = [o for o in self.inbox if o not in ready]
        return ready

    def acknowledge_order(self, order: Order) -> bool:
        self.acknowledged.append(order.external_id)
        return True

    def submit_tracking(self, order: Order, shipment: Shipment) -> bool:
        self.tracking[order.external_id] = shipment
        return True

    def seed_order(self, order: Order) -> None:
        """Drop an order into the inbox as though a customer had placed it."""
        self.inbox.append(order)


# -- Shopify ---------------------------------------------------------------

SHOPIFY_API_VERSION = "2025-01"

_FULFILLMENT_ORDERS_QUERY = """
query orderFulfillmentOrders($id: ID!) {
  order(id: $id) {
    id
    name
    fulfillmentOrders(first: 10) {
      nodes {
        id
        status
        lineItems(first: 50) { nodes { id remainingQuantity } }
      }
    }
  }
}
"""

_FULFILLMENT_CREATE_MUTATION = """
mutation fulfillmentCreate($fulfillment: FulfillmentInput!) {
  fulfillmentCreate(fulfillment: $fulfillment) {
    fulfillment { id status trackingInfo(first: 10) { company number url } }
    userErrors { field message }
  }
}
"""


@dataclass
class ShopifyConnector(ChannelConnector):
    """Live Shopify adapter over the GraphQL Admin API.

    The fulfilment path is the part worth spelling out, because it is not
    obvious from the outside. Shipping an order is two calls, not one:

    1. Read the order's ``fulfillmentOrders`` -- Shopify's unit of fulfilment
       work, one per assigned location -- and its remaining line items.
    2. Call ``fulfillmentCreate`` with ``lineItemsByFulfillmentOrder`` and the
       ``trackingInfo``. Later corrections go through
       ``fulfillmentTrackingInfoUpdate`` instead.

    One constraint that bites dropshippers specifically: since API version
    2024-10, ``fulfillmentCreate`` only works for orders assigned to a
    merchant-managed location or to a fulfilment service you own. An order
    routed to somebody else's third-party fulfilment service will reject, and
    the fix is ``fulfillmentOrderSubmitFulfillmentRequest`` instead.
    """

    shop_domain: str = ""
    channel: Channel = Channel.SHOPIFY
    token_ref: SecretRef = field(default_factory=lambda: SecretRef("SHOPIFY_ACCESS_TOKEN"))
    resolver: SecretResolver = field(default_factory=SecretResolver)
    limiter: RateLimiter = field(default_factory=lambda: RateLimiter(capacity=4, refill_per_second=2.0))
    guard: IdempotencyGuard = field(default_factory=IdempotencyGuard)
    timeout: float = 20.0

    def is_live(self) -> bool:
        return bool(self.shop_domain) and self.resolver.has(self.token_ref)

    @property
    def endpoint(self) -> str:
        return f"https://{self.shop_domain}/admin/api/{SHOPIFY_API_VERSION}/graphql.json"

    def execute(self, query: str, variables: dict | None = None) -> dict:
        """POST a GraphQL document and return ``data``, raising on any error.

        GraphQL answers 200 for business-logic failures, so both the transport
        ``errors`` array and every mutation's ``userErrors`` have to be checked
        explicitly. Treating a 200 as success is the classic way to believe a
        fulfilment succeeded when it did not.
        """
        if not self.shop_domain:
            raise ChannelError("ShopifyConnector needs shop_domain, e.g. my-shop.myshopify.com")
        if not self.limiter.allow():
            raise ChannelError(
                f"Shopify rate limit reached; retry in {self.limiter.wait_time():.1f}s"
            )
        body = json.dumps({"query": query, "variables": variables or {}}).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-Shopify-Access-Token": self.resolver.resolve(self.token_ref),
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - needs a live shop
            raise ChannelError(f"Shopify returned HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:  # pragma: no cover - needs a network
            raise ChannelError(f"could not reach Shopify: {exc.reason}") from exc

        if payload.get("errors"):
            raise ChannelError(f"Shopify GraphQL errors: {payload['errors']}")
        return payload.get("data") or {}

    @staticmethod
    def _raise_user_errors(node: dict, action: str) -> None:
        errors = node.get("userErrors") or []
        if errors:
            detail = "; ".join(f"{e.get('field')}: {e.get('message')}" for e in errors)
            raise ChannelError(f"Shopify rejected {action}: {detail}")

    def publish_listing(self, listing: Listing) -> str:  # pragma: no cover - needs a live shop
        data = self.execute(
            """
            mutation productCreate($input: ProductInput!) {
              productCreate(input: $input) {
                product { id }
                userErrors { field message }
              }
            }
            """,
            {
                "input": {
                    "title": listing.title,
                    "descriptionHtml": listing.description,
                    "status": "ACTIVE",
                }
            },
        )
        node = data.get("productCreate") or {}
        self._raise_user_errors(node, "productCreate")
        external_id = (node.get("product") or {}).get("id", "")
        listing.external_id = external_id
        listing.state = ListingState.PUBLISHED
        return external_id

    def update_inventory(self, sku: str, quantity: int) -> bool:  # pragma: no cover
        raise NotImplementedError(
            "Inventory needs the location's inventoryItem id: query productVariants "
            "for inventoryItem.id, then call inventorySetQuantities with that "
            "location. Wire it once you know which location holds your stock."
        )

    def fetch_orders(self, since: datetime | None = None) -> list[Order]:  # pragma: no cover
        raise NotImplementedError(
            "Map Shopify's order payload onto domain.Order here. Prefer the "
            "orders/create webhook over polling, and verify its HMAC with "
            "security.verify_webhook before trusting it."
        )

    def acknowledge_order(self, order: Order) -> bool:  # pragma: no cover
        return True  # Shopify has no separate acknowledgement step.

    def submit_tracking(self, order: Order, shipment: Shipment) -> bool:  # pragma: no cover
        """Fulfil the order and attach tracking, exactly once per shipment."""
        key = idempotency_key("shopify.fulfil", order.external_id, shipment.tracking_number)
        if not self.guard.claim(key):
            return bool(self.guard.result_for(key))

        data = self.execute(_FULFILLMENT_ORDERS_QUERY, {"id": order.external_id})
        nodes = (((data.get("order") or {}).get("fulfillmentOrders") or {}).get("nodes")) or []
        open_orders = [n for n in nodes if n.get("status") in ("OPEN", "IN_PROGRESS")]
        if not open_orders:
            raise ChannelError(f"no open fulfillment orders on {order.external_id}")

        line_items = [
            {
                "fulfillmentOrderId": node["id"],
                "fulfillmentOrderLineItems": [
                    {"id": li["id"], "quantity": li["remainingQuantity"]}
                    for li in (node.get("lineItems") or {}).get("nodes", [])
                    if li.get("remainingQuantity", 0) > 0
                ],
            }
            for node in open_orders
        ]
        result = self.execute(
            _FULFILLMENT_CREATE_MUTATION,
            {
                "fulfillment": {
                    "lineItemsByFulfillmentOrder": line_items,
                    "notifyCustomer": True,
                    "trackingInfo": {
                        "company": shipment.carrier,
                        "number": shipment.tracking_number,
                        "url": shipment.tracking_url,
                    },
                }
            },
        )
        node = result.get("fulfillmentCreate") or {}
        self._raise_user_errors(node, "fulfillmentCreate")
        succeeded = (node.get("fulfillment") or {}).get("status") == "SUCCESS"
        self.guard.record(key, succeeded)
        return succeeded


# -- Amazon and eBay -------------------------------------------------------


@dataclass
class _UnwiredConnector(ChannelConnector):
    """Shared body for adapters that need an auth flow before they can work."""

    channel: Channel = Channel.AMAZON
    service: str = "the marketplace API"
    guidance: str = ""

    def _unwired(self, verb: str) -> "NotImplementedError":
        return NotImplementedError(
            f"{self.channel.label} {verb} is not wired up. {self.guidance} "
            "Until then this channel runs against InMemoryChannel, which "
            "implements the full contract."
        )

    def publish_listing(self, listing: Listing) -> str:
        raise self._unwired("listing publication")

    def update_inventory(self, sku: str, quantity: int) -> bool:
        raise self._unwired("inventory sync")

    def fetch_orders(self, since: datetime | None = None) -> list[Order]:
        raise self._unwired("order polling")

    def acknowledge_order(self, order: Order) -> bool:
        raise self._unwired("order acknowledgement")

    def submit_tracking(self, order: Order, shipment: Shipment) -> bool:
        raise self._unwired("tracking submission")


def amazon_connector() -> _UnwiredConnector:
    """Amazon Selling Partner API port."""
    return _UnwiredConnector(
        channel=Channel.AMAZON,
        service="SP-API",
        guidance=(
            "SP-API needs Login with Amazon: exchange a refresh token for an "
            "access token, then use the Orders and Feeds APIs. Listings go via "
            "the JSON_LISTINGS_FEED, shipping confirmation via "
            "POST_ORDER_FULFILLMENT_DATA. Note that Amazon's drop shipping "
            "policy requires you to be the seller of record on every packing "
            "slip -- policy.py blocks non-compliant candidates before they "
            "reach this connector."
        ),
    )


def ebay_connector() -> _UnwiredConnector:
    """eBay Sell API port."""
    return _UnwiredConnector(
        channel=Channel.EBAY,
        service="Sell API",
        guidance=(
            "eBay needs an OAuth user token. Listings go through the Inventory "
            "API (createOrReplaceInventoryItem, then publishOffer); orders and "
            "shipping come from the Fulfillment API "
            "(getOrders, createShippingFulfillment)."
        ),
    )


def connector_for(channel: Channel, live: bool = False, **kwargs) -> ChannelConnector:
    """Pick an adapter for ``channel``.

    Defaults to the in-memory adapter. Going live is an explicit, per-channel
    decision -- there is no environment variable that silently promotes a demo
    run into one that spends money.
    """
    if not live:
        return InMemoryChannel(channel=channel)
    if channel is Channel.SHOPIFY:
        return ShopifyConnector(**kwargs)
    if channel is Channel.AMAZON:
        return amazon_connector()
    return ebay_connector()
