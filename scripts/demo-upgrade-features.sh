#!/bin/bash
# Demo script showing the new upgrade capabilities
# Usage: ./scripts/demo-upgrade-features.sh

echo "🚀 Nuro-Padel Upgrade Features Demo"
echo "=================================="

# Check if services are running
echo ""
echo "📋 Step 1: Check Current Service Status"
echo "---------------------------------------"

echo "Checking YOLO Combined service health..."
curl -s http://localhost:8001/healthz | jq '.service, .models, .features' || echo "❌ YOLO Combined not accessible"

echo ""
echo "Checking MMPose service health..."
curl -s http://localhost:8003/healthz | jq '.service, .models, .features' || echo "❌ MMPose not accessible"

echo ""
echo "Checking YOLO-NAS service health..."
curl -s http://localhost:8004/healthz | jq '.service, .models, .features' || echo "❌ YOLO-NAS not accessible"

echo ""
echo "📝 Step 2: Current Model Configuration"
echo "-------------------------------------"
echo "YOLO Combined models:"
cat services/yolo-combined/config/model_config.json | jq '.models'

echo ""
echo "🎛️ Step 3: Toggle a Feature Flag"
echo "--------------------------------"
echo "Current TrackNet v4 status in YOLO Combined:"
cat services/yolo-combined/config/model_config.json | jq '.features.tracknet_v4'

echo ""
echo "Let's enable TrackNet v4 by editing the config file..."
# Create a backup and enable tracknet_v4
cp services/yolo-combined/config/model_config.json services/yolo-combined/config/model_config.json.backup

# Use jq to enable tracknet_v4
jq '.features.tracknet_v4.enabled = true' services/yolo-combined/config/model_config.json.backup > services/yolo-combined/config/model_config.json

echo "✅ TrackNet v4 enabled in config file"
echo "New status:"
cat services/yolo-combined/config/model_config.json | jq '.features.tracknet_v4'

echo ""
echo "📡 Step 4: Check if Config Hot-Reloaded"
echo "---------------------------------------"
echo "Checking health endpoint for updated feature flags..."
sleep 2  # Give it a moment to reload
curl -s http://localhost:8001/healthz | jq '.features.tracknet_v4' || echo "❌ Service not accessible"

echo ""
echo "🔧 Step 5: Environment Variable Override Demo"
echo "---------------------------------------------"
echo "Current enhanced ball tracking feature:"
curl -s http://localhost:8001/healthz | jq '.features.enhanced_ball_tracking' || echo "❌ Service not accessible"

echo ""
echo "Setting environment variable to disable enhanced ball tracking..."
export FEATURE_ENHANCED_BALL_TRACKING_ENABLED=false

echo "Restart the service to see environment override take effect:"
echo "docker-compose restart yolo-combined"

echo ""
echo "🔄 Step 6: Restore Original Configuration"
echo "-----------------------------------------"
echo "Restoring backup configuration..."
cp services/yolo-combined/config/model_config.json.backup services/yolo-combined/config/model_config.json
rm services/yolo-combined/config/model_config.json.backup

echo "✅ Configuration restored"

echo ""
echo "📊 Step 7: Full Health Check Summary"
echo "------------------------------------"
echo "Getting comprehensive status from all services..."

echo ""
echo "🚀 YOLO Combined Service:"
curl -s http://localhost:8001/healthz | jq '{
  status: .status,
  service_version: .service.version,
  models_loaded: [.models | to_entries[] | select(.value.loaded == true) | .key],
  enabled_features: [.features | to_entries[] | select(.value.enabled == true) | .key],
  deployment: .deployment
}' || echo "❌ Not accessible"

echo ""
echo "🤸 MMPose Service:"
curl -s http://localhost:8003/healthz | jq '{
  status: .status,
  service_version: .service.version,
  models_loaded: [.models | to_entries[] | select(.value.loaded == true) | .key],
  enabled_features: [.features | to_entries[] | select(.value.enabled == true) | .key],
  biomechanics_enabled: .biomechanics
}' || echo "❌ Not accessible"

echo ""
echo "🎯 YOLO-NAS Service:"
curl -s http://localhost:8004/healthz | jq '{
  status: .status,
  service_version: .service.version,
  models_loaded: [.models | to_entries[] | select(.value.loaded == true) | .key],
  enabled_features: [.features | to_entries[] | select(.value.enabled == true) | .key],
  optimization: .optimization
}' || echo "❌ Not accessible"

echo ""
echo "✅ Demo Complete!"
echo "================"
echo ""
echo "🎉 What we demonstrated:"
echo "• Model version visibility in health checks"
echo "• Feature flag configuration and hot-reloading"
echo "• Environment variable overrides"
echo "• Comprehensive service status monitoring"
echo ""
echo "🚀 Next steps for upgrades:"
echo "• Change model files in config/*.json"
echo "• Toggle features without code changes"
echo "• Use environment variables for quick overrides"
echo "• Monitor upgrade readiness via /healthz endpoints"