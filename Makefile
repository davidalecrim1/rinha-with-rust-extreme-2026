DOCKER_IMAGE := davidalecrim1/rinha-rust-2026

.PHONY: lint run test release load-test

lint:
	cargo fmt --check
	cargo clippy -- -D warnings

test:
	cargo test

run:
	docker compose -f docker-compose.local.yml up --build -d

load-test: run
	@echo "Waiting for stack to be ready..."
	@until curl -sf http://localhost:9999/ready >/dev/null 2>&1; do sleep 1; done
	@echo "Stack ready. Running load test..."
	$(eval CURRENT_TAG := $(shell git tag --sort=-v:refname | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$$' | head -1))
	@mkdir -p scripts/results
	k6 run --env VERSION=$(CURRENT_TAG) scripts/load-test.js

# Increments the patch version, tags, pushes to git, builds the linux/amd64
# release image, publishes it to Docker Hub, and updates the submission branch
# (docker-compose.yml, nginx.conf, info.json) with the new image tag.
release:
	$(eval LAST_TAG := $(shell git tag --sort=-v:refname | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$$' | head -1))
	$(eval VERSION := $(if $(LAST_TAG),$(shell echo $(LAST_TAG) | awk -F. '{print $$1"."$$2"."$$3+1}'),v0.1.0))
	@echo "Releasing $(VERSION)"
	git tag $(VERSION)
	git push origin $(VERSION)
	docker build --platform linux/amd64 -t $(DOCKER_IMAGE):$(VERSION) -t $(DOCKER_IMAGE):latest .
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
	@cp nginx.conf info.json /tmp/rinha-submission/
	@cd /tmp/rinha-submission && git add -A && git commit -m "release $(VERSION)" && git push -u origin submission
	@git worktree remove /tmp/rinha-submission
	@echo "Released $(VERSION) — submission branch updated"
	@echo ""
	@echo "Don't forget: add an entry for $(VERSION) to CHANGELOG.md (max 3 bullet points)"
