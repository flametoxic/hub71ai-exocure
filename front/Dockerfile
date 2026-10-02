FROM node:22-alpine AS build
WORKDIR /app
ENV ELECTRON_SKIP_BINARY_DOWNLOAD=1
COPY package.json package-lock.json ./
RUN npm ci
COPY vite.config.mjs vite.web.config.mjs ./
COPY backend-contract.mjs ./
COPY electron/renderer ./electron/renderer
COPY electron/logo.svg electron/logocolor.svg ./electron/
RUN npm run build:web

FROM nginx:stable-alpine
ENV CURE_API_UPSTREAM=http://host.docker.internal:8000
ENV NGINX_ENVSUBST_FILTER=^CURE_API_UPSTREAM$
COPY nginx/default.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist/web /usr/share/nginx/html
EXPOSE 80
