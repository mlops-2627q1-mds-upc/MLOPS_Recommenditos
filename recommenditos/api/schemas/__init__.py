"""The request and response bodies of the API.

`common` holds what every body shares: the field types, `ListingFields`, the
model description, warnings and the error shape. Every other module holds the
bodies of one endpoint and is named after its route module in `routes/`, so
endpoints added in parallel do not edit the same file.
"""
