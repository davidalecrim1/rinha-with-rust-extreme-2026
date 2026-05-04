DOCKER_IMAGE := davidalecrim1/rinha-rust-extreme-2026

.PHONY: lint build run test release load-test official-load-test profile fetch-test-data

lint:
	cargo fmt --check
	cargo clippy -- -D warnings

test:
	cargo test
	rustup target add x86_64-apple-darwin
	cargo test --target x86_64-apple-darwin

build:
	docker build --platform linux/amd64 -t $(DOCKER_IMAGE):latest .

run:
	docker compose -f docker-compose.yml up -d

fetch-test-data:
	cp ../rinha-de-backend-2026/test/test-data.json scripts/test-data.json

load-test:
	docker compose -f docker-compose.local.yml up --build -d
	@echo "Waiting for stack to be ready..."
	@until curl -sf http://localhost:9999/ready >/dev/null 2>&1; do sleep 1; done
	@echo "Stack ready. Running load test..."
	$(eval CURRENT_TAG := $(shell git tag --sort=-v:refname | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$$' | head -1))
	@mkdir -p scripts/results
	k6 run --env VERSION=$(CURRENT_TAG) scripts/load-test.js

official-load-test:
	docker compose -f docker-compose.local.yml up --build -d
	@echo "Waiting for stack to be ready..."
	@until curl -sf http://localhost:9999/ready >/dev/null 2>&1; do sleep 1; done
	@echo "Stack ready. Running official load test..."
	cd ../rinha-de-backend-2026 && k6 run test/test.js

profile:
	@mkdir -p profile-api1 profile-api2
	docker compose -f docker-compose.local.yml -f docker-compose.profile.yml up --build -d
	@echo "Waiting for stack to be ready..."
	@until curl -sf http://localhost:9999/ready >/dev/null 2>&1; do sleep 1; done
	@echo "Stack ready. Running load test..."
	$(eval CURRENT_TAG := $(shell git tag --sort=-v:refname | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$$' | head -1))
	@mkdir -p scripts/results
	k6 run --env VERSION=$(CURRENT_TAG) scripts/load-test.js
	@echo "Stopping containers to flush profile data..."
	docker compose -f docker-compose.local.yml -f docker-compose.profile.yml stop api1 api2
	@echo ""
	@echo "=== Profile data ==="
	@echo "Collapsed stacks: profile-api1/profile.folded, profile-api2/profile.folded"
	@echo "Flamegraph SVGs:  profile-api1/profile.svg, profile-api2/profile.svg"

# Tags and publishes the latest locally built linux/amd64 image, then updates the
# submission branch. Pass VERSION explicitly to control the tag:
#   make release VERSION=v0.7.0
# When VERSION is not set, the patch of the latest tag is incremented.
# Run `make build` first when source changes need to be included.
release:
	$(eval LAST_TAG := $(shell git tag --sort=-v:refname | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$$' | head -1))
	$(eval VERSION := $(or $(VERSION),$(if $(LAST_TAG),$(shell echo $(LAST_TAG) | awk -F. '{print $$1"."$$2"."$$3+1}'),v0.1.0)))
	@echo "Releasing $(VERSION)"
	@docker image inspect $(DOCKER_IMAGE):latest >/dev/null || { echo "Missing $(DOCKER_IMAGE):latest. Run make build first."; exit 1; }
	git tag $(VERSION)
	git push origin $(VERSION)
	docker tag $(DOCKER_IMAGE):latest $(DOCKER_IMAGE):$(VERSION)
	docker push $(DOCKER_IMAGE):$(VERSION)
	docker push $(DOCKER_IMAGE):latest
	@echo "Updating submission branch..."
	@git worktree remove --force /tmp/rinha-submission 2>/dev/null || true
	@if git show-ref --verify --quiet refs/heads/submission; then \
		git worktree add /tmp/rinha-submission submission; \
	elif git ls-remote --exit-code origin submission >/dev/null 2>&1; then \
		git fetch origin submission:submission && git worktree add /tmp/rinha-submission submission; \
	else \
		git worktree add --orphan -b submission /tmp/rinha-submission; \
	fi
	@sed 's|$(DOCKER_IMAGE):latest|$(DOCKER_IMAGE):$(VERSION)|g' docker-compose.yml > /tmp/rinha-submission/docker-compose.yml
	@cp docker-compose.local.yml docker-compose.profile.yml nginx.conf info.json /tmp/rinha-submission/
	@cd /tmp/rinha-submission && git add -A && git commit -m "release $(VERSION)" && git push -u origin submission
	@git worktree remove /tmp/rinha-submission
	@echo "Released $(VERSION) — submission branch updated"
	@echo ""
	@echo "Don't forget: add an entry for $(VERSION) to CHANGELOG.md (max 3 bullet points)"
