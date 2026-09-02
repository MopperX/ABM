# Routing contract

Application routes are normalized for separators only. Route **case is significant** and must be preserved because several upstream systems use case-sensitive path segments.

Rules:
1. trim surrounding whitespace;
2. ensure exactly one leading slash;
3. collapse repeated slashes;
4. remove a trailing slash except for `/`;
5. never lowercase or uppercase route content.
