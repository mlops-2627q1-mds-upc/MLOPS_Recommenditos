"""The request and response bodies of the API.

`common` holds what every body shares: the field types, `ListingFields`, the
model description, warnings and the error shape. Every other module holds the
bodies of one endpoint and is named after its route module in `routes/`, so
endpoints added in parallel do not edit the same file. A request body that
several endpoints take has a module of its own: `partial_listing` is the input
of `/v1/price-range` and of `/v1/comparables` once it exists (#138).
"""
