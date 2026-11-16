# AIORI website: builds the final frontend (with Replay and the Disruption Explorer) and serves it with nginx,
# which forwards /api/ to the backend on the internal network.
FROM node:22-alpine AS build
WORKDIR /app
COPY src/app/web-phase10/package.json src/app/web-phase10/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY src/app/web-phase10/ ./
# Same steps as `npm run build:phase10`; the API base stays empty, so every call is relative (/api/...).
RUN npx tsc --noEmit && npx vite build --mode replay --outDir dist

FROM nginxinc/nginx-unprivileged:1.27-alpine
COPY deploy/aiori/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 8080
