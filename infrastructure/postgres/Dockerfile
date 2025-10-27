FROM postgres:15

# Install pg_cron extension for PostgreSQL 15
RUN apt-get update \
    && apt-get install -y --no-install-recommends postgresql-15-cron \
    && rm -rf /var/lib/apt/lists/*

# Expose default Postgres port (documentative)
EXPOSE 5432


