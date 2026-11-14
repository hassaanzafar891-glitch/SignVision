FROM node:22-bookworm-slim AS build
WORKDIR /app/frontend
# better-sqlite3 may need node-gyp when a prebuilt binary is unavailable.
RUN apt-get update && apt-get install -y --no-install-recommends python3 make g++ \
    && rm -rf /var/lib/apt/lists/*
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
# The existing SQLite module stores data in ../backend/data.
RUN mkdir -p /app/backend/data
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

FROM node:22-bookworm-slim AS runtime
WORKDIR /app/frontend
ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1
COPY --from=build --chown=node:node /app/frontend ./
RUN mkdir -p /app/backend/data && chown -R node:node /app/backend
USER node
EXPOSE 3000
CMD ["npm", "run", "start", "--", "--hostname", "0.0.0.0"]
