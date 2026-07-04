#!/usr/bin/bash
#Load environment
source .env

# Create timescaledb shared folder
mkdir -p ./data/timescaledb
echo "Starting up container"
# Start container for TimeScaleDb
docker compose up -d
echo "Container started"

# Wait for TimeScaleDb to be ready
echo "Waiting for TimescaleDB to be ready..."
until docker exec timescaledb pg_isready -U postgres > /dev/null 2>&1; do
	sleep 1
done
echo "TimescaleDB is ready"

# Create table
echo "Setup the database"
docker exec -i timescaledb psql -U postgres -d "${POSTGRES_DB}" << EOF
CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS price_candles (
	time        TIMESTAMPTZ      NOT NULL,
	symbol      TEXT             NOT NULL,
	open 	    DOUBLE PRECISION NOT NULL,
	high        DOUBLE PRECISION NOT NULL,
	low         DOUBLE PRECISION NOT NULL,
	close       DOUBLE PRECISION NOT NULL,
	volume      BIGINT           NOT NULL,
	UNIQUE (time, symbol)
);

SELECT create_hypertable('price_candles', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS price_candles_symbol_time ON price_candles (symbol, time DESC);
EOF
echo "Setup complete"
