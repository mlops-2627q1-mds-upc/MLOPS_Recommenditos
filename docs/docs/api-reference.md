# API reference

The OpenAPI document of the price API, rendered from [`openapi.json`](openapi.json), the copy committed to the repository.
It is the contract the service serves at `/openapi.json`, readable here without running the service; [The API](api.md) says how to run it and how to change the contract.

The checked listing fields (`make`, `fuel_category` and the other categorical fields) are typed as strings here.
The values they accept come from the model bundle the service loads, so the live `/docs` of a running service lists them, and a 422 names them for a refused value ([Errors](api.md#errors)).

<swagger-ui src="openapi.json"/>
