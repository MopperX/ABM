# Borealis backend order

The Borealis service has two production backends: alpha and bravo. Alpha must be drained and deployed first. Bravo is deployed only after alpha has passed its health check and rejoined the load balancer.
