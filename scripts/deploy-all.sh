#!/bin/bash

echo "🚀 Deploying all NuroPadel services..."

# Define services with their ports for reference
declare -A SERVICES
SERVICES["yolo8"]="8001"
SERVICES["yolo11"]="8007"
SERVICES["mmpose"]="8003"
SERVICES["vitpose"]="8004"
SERVICES["rf-detr"]="8005"
SERVICES["yolo-nas"]="8006"

FAILED_SERVICES=()
DEPLOYED_SERVICES=()

# Change to project root directory
cd "$(dirname "$0")/.."

echo "📋 Services to deploy:"
for service in "${!SERVICES[@]}"; do
    echo "  - $service (port ${SERVICES[$service]})"
done
echo ""

# Deploy each service
for service in "${!SERVICES[@]}"; do
    echo "📦 Deploying $service..."
    
    if [ -f "services/$service/deploy.sh" ]; then
        cd "services/$service"
        
        # Make script executable
        chmod +x deploy.sh
        
        if ./deploy.sh; then
            echo "✅ $service deployed successfully"
            DEPLOYED_SERVICES+=("$service")
        else
            echo "❌ $service deployment failed"
            FAILED_SERVICES+=("$service")
        fi
        
        cd "../.."
    else
        echo "⚠️  Deploy script not found for $service"
        FAILED_SERVICES+=("$service")
    fi
    
    echo ""
done

# Summary
echo "🎯 Deployment Summary:"
echo "====================="

if [ ${#DEPLOYED_SERVICES[@]} -gt 0 ]; then
    echo "✅ Successfully deployed services:"
    for service in "${DEPLOYED_SERVICES[@]}"; do
        echo "  - $service → http://35.189.53.46:${SERVICES[$service]}"
    done
fi

if [ ${#FAILED_SERVICES[@]} -gt 0 ]; then
    echo "❌ Failed services: ${FAILED_SERVICES[*]}"
    echo "💡 Other services are still running normally"
    echo ""
    echo "🔧 To retry failed services:"
    for service in "${FAILED_SERVICES[@]}"; do
        echo "  ./services/$service/deploy.sh"
    done
    exit 1
else
    echo "🎉 All services deployed successfully!"
    echo ""
    echo "🔗 Service Endpoints:"
    for service in "${!SERVICES[@]}"; do
        echo "  - $service: http://35.189.53.46:${SERVICES[$service]}/healthz"
    done
fi