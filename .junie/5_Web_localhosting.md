# Perudo Web Frontend with Local Hosting Specification

This document outlines the development plan for creating a multiplayer web frontend for the Perudo game with local hosting capabilities, following Test-Driven Development (TDD) principles.

## Overview

The goal is to create a comprehensive web-based multiplayer interface that leverages the existing sophisticated AI agents, particularly the full-featured Monte Carlo agent, while providing real-time multiplayer functionality through modern web technologies.

### Current State Analysis

The existing codebase provides excellent foundations for web frontend development:
- **Robust Game Engine**: `sim/perudo.py` with comprehensive state management and action handling
- **Sophisticated AI Agents**: Random, Baseline, and Monte Carlo agents with advanced features
- **Comprehensive Testing**: 132 tests following TDD principles ensuring reliability
- **Advanced Features**: Betting history, trust management, parallel processing capabilities
- **Modular Architecture**: Clean separation of concerns prepared for extension

### Target User Scenarios

1. **Room Creation**: Choose nickname, create room with adjustable settings (player count, bot configuration)
2. **Room Joining**: Join existing rooms via generated invite codes
3. **Pre-game Setup**: Player seat arrangement, bot difficulty selection
4. **Real-time Gameplay**: 60-second action timers, dice visibility controls, live updates
5. **Game Results**: Round-by-round dice reveals, end-game statistics, restart functionality

## Technical Architecture

### Backend Technology Stack

**Core Framework**: FastAPI (Python 3.8+) for seamless integration with existing codebase
**Real-time Communication**: WebSocket for multiplayer synchronization
**Database**: SQLite for local persistence and room management
**AI Integration**: Direct usage of existing agent implementations without modification
**External Access**: Cloudflare Tunnel for public accessibility without port forwarding

### Frontend Technology Stack

**Framework**: React 18 with TypeScript for component-based architecture
**State Management**: React Context with useReducer for game state
**Styling**: Tailwind CSS for responsive design
**Real-time Client**: Native WebSocket API with reconnection handling
**Build Tool**: Vite for fast development and optimized builds

### Key Architectural Decisions

**Local Hosting Benefits**:
- Zero computational limitations for Monte Carlo agent (full n=200+ simulations)
- Complete access to advanced AI features (parallel processing, Bayesian sampling)
- No monthly hosting costs or service restrictions
- Instant deployment and real-time debugging capabilities

**Network Access Strategy**:
- Cloudflare Tunnel for secure external access without complex router configuration
- Alternative ngrok support for quick testing scenarios
- HTTPS automatically provided through tunnel services

## Core Components Implementation

### Game Room Management

```python
# Room data model with comprehensive settings
@dataclass
class RoomSettings:
    max_players: int = 6
    bot_count: int = 0
    bot_types: List[str] = None  # ['mc', 'baseline', 'random']
    bot_difficulties: List[str] = None  # ['easy', 'medium', 'hard']
    start_dice: int = 5
    timer_duration: int = 60
    ones_wild: bool = True
    maputa: bool = True
    exact: bool = True
```

**Room Code Generation**: Cryptographically secure 6-character codes using uppercase letters and numbers
**Bot Configuration**: Support for all existing AI agents with difficulty levels mapped to simulation parameters
**Player Management**: Session-based tracking with reconnection support

### Real-Time Communication Protocol

**WebSocket Message Structure**:
```typescript
interface WebSocketMessage {
  type: string;
  data: any;
  timestamp: number;
  messageId?: string;
}
```

**Core Message Types**:
- `room_join/room_update`: Player management and room status
- `player_action/game_state_update`: Game actions and state synchronization  
- `timer_update/timer_expired`: Action timing and auto-forfeit handling
- `round_result/game_end`: Round outcomes and victory conditions

### Timer Management System

**ActionTimer Class**: Asynchronous countdown with periodic updates and auto-forfeit capability
**TimerManager**: Centralized timer coordination for multiple concurrent games
**Bot Integration**: Immediate processing for AI agents with realistic delay simulation

### Monte Carlo Agent Integration

**Full Feature Support**: Complete access to all advanced MC agent capabilities without restrictions
```python
PRODUCTION_MC_CONFIG = {
    'n': 300,                           # High simulation count
    'enable_parallel': True,            # Multi-core processing  
    'weighted_sampling': True,          # History-aware determinization
    'enhanced_pruning': True,           # Advanced action filtering
    'variance_reduction': True,         # Statistical optimization
    'betting_history_enabled': True,    # Player modeling
    'bayesian_sampling': True,          # Bayesian inference
    'player_trust_enabled': True       # Trust management
}
```

**Difficulty Scaling**: Configurable simulation counts and feature enablement based on selected difficulty
**Async Processing**: Non-blocking AI computation with progress updates for longer computations

## Database Schema

### SQLite Tables for Local Persistence

```sql
-- Room persistence with settings and status tracking
CREATE TABLE rooms (
    code VARCHAR(6) PRIMARY KEY,
    host_id VARCHAR(36) NOT NULL,
    settings TEXT NOT NULL,  -- JSON configuration
    status VARCHAR(20) DEFAULT 'waiting',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Player session management for reconnection support
CREATE TABLE player_sessions (
    session_id VARCHAR(36) PRIMARY KEY,
    room_code VARCHAR(6),
    nickname VARCHAR(50) NOT NULL,
    player_order INTEGER,
    is_bot BOOLEAN DEFAULT FALSE,
    bot_type VARCHAR(20),
    bot_difficulty VARCHAR(10),
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (room_code) REFERENCES rooms(code)
);

-- Game state snapshots for crash recovery
CREATE TABLE game_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    room_code VARCHAR(6),
    game_state TEXT NOT NULL,  -- JSON serialized state
    round_number INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (room_code) REFERENCES rooms(code)
);
```

## Performance Optimizations

### Local Hosting Resource Management

**Concurrent Game Limits**: Maximum 3 simultaneous games to ensure responsive performance
**Memory Management**: Efficient WebSocket connection pooling and message batching
**CPU Optimization**: Smart bot distribution (max 2 MC agents per game) and adaptive difficulty

### WebSocket Performance

**Connection Management**: 
- Ping/pong keepalive (20s intervals)
- Graceful reconnection with state recovery
- Message compression for reduced bandwidth

**Message Batching**: 
- Queue non-critical updates for batch delivery
- Immediate delivery for critical game actions
- Configurable batch intervals (default 100ms)

## Development Implementation Plan

### Phase 1: Foundation Setup (Weeks 1-2)

**Backend Infrastructure**:
- FastAPI application with WebSocket endpoint configuration
- SQLite database schema implementation and migrations
- Basic room creation and joining functionality
- Integration with existing `sim/perudo.py` game engine

**Frontend Foundation**:
- React application setup with TypeScript configuration
- Basic routing and component structure
- Room creation and joining UI components
- WebSocket connection management hooks

**Testing Requirements**:
- Unit tests for room management logic
- WebSocket connection integration tests
- Database operation validation tests
- Basic UI component rendering tests

### Phase 2: Core Game Integration (Weeks 3-4)

**Game Service Implementation**:
- Complete integration with `PerudoSimulator` class
- AI agent factory with configuration mapping
- Timer service with async countdown functionality
- Game state serialization and recovery

**Real-Time Communication**:
- Complete WebSocket message protocol implementation
- Game state synchronization between clients
- Player action validation and processing
- Error handling and recovery mechanisms

**Testing Requirements**:
- Full game flow integration tests
- AI agent integration validation
- Timer functionality and auto-forfeit testing
- Message protocol compliance verification

### Phase 3: Complete User Interface (Weeks 5-6)

**Game Interface Components**:
- Interactive dice display with visibility controls
- Betting interface with legal action validation
- Player status panel with dice counts and timers
- Action history display and round result presentations

**Advanced Features**:
- Player seat swapping functionality
- Bot difficulty selection interface
- Game settings configuration panel
- Responsive mobile-friendly design

**Testing Requirements**:
- UI component interaction testing
- Game interface usability validation
- Mobile responsiveness verification
- Cross-browser compatibility testing

### Phase 4: Deployment and Optimization (Weeks 7-8)

**Local Hosting Setup**:
- Cloudflare Tunnel configuration and documentation
- Production deployment scripts and monitoring
- Performance optimization and resource management
- Security hardening and input validation

**Final Integration**:
- End-to-end testing with multiple concurrent games
- Performance benchmarking under realistic load
- Documentation and user guide creation
- Error monitoring and logging implementation

**Testing Requirements**:
- Load testing with multiple concurrent rooms
- Network connectivity and reconnection testing
- Security vulnerability assessment
- Performance benchmarking against targets

## Configuration Parameters

### Server Configuration

| Parameter | Default | Description | Purpose |
|-----------|---------|-------------|---------|
| `host` | "0.0.0.0" | Server bind address | External access |
| `port` | 8000 | HTTP/WebSocket port | Service endpoint |
| `max_concurrent_rooms` | 3 | Room limit | Resource management |
| `max_players_per_room` | 6 | Player limit | Game balance |
| `room_expiry_hours` | 24 | Room cleanup | Storage management |

### Game Configuration

| Parameter | Default | Description | Purpose |
|-----------|---------|-------------|---------|
| `default_timer` | 60 | Action timeout (seconds) | Game pacing |
| `max_mc_agents_per_room` | 2 | MC agent limit | Performance |
| `enable_reconnection` | True | Session recovery | User experience |
| `snapshot_frequency` | 5 | State backup interval | Crash recovery |

## Security and Validation

### Input Validation

**Room Code Security**: Cryptographically secure random generation with collision detection
**Action Validation**: Server-side verification of all game actions against legal move sets
**Rate Limiting**: Request throttling to prevent abuse (60 requests/minute per IP)
**Session Management**: Secure session tokens with configurable expiration

### Network Security

**HTTPS Enforcement**: Automatic SSL/TLS through Cloudflare Tunnel
**CORS Configuration**: Restricted cross-origin access for security
**WebSocket Authentication**: Token-based connection validation
**Input Sanitization**: Comprehensive validation of all user inputs

## Monitoring and Logging

### Performance Metrics

**Game Performance**: Monte Carlo agent response times and decision quality metrics
**System Performance**: Memory usage, CPU utilization, and concurrent connection counts
**Network Performance**: WebSocket message latency and connection stability
**User Experience**: Game completion rates and error frequency

### Logging Strategy

**Structured Logging**: JSON-formatted logs with correlation IDs for request tracing
**Error Tracking**: Comprehensive exception logging with stack traces
**Audit Trail**: User actions and game events for debugging and analysis
**Performance Logging**: Response time tracking and resource utilization monitoring

## Testing Strategy

### Test-Driven Development Approach

**Unit Tests**: Individual component functionality validation
- Room management operations
- Game state transitions  
- AI agent integration
- Timer and WebSocket functionality

**Integration Tests**: Component interaction validation
- Full game flow scenarios
- Multi-player synchronization
- Bot integration testing
- Database persistence verification

**End-to-End Tests**: Complete user scenario validation
- Room creation to game completion workflows
- Error recovery and reconnection scenarios
- Performance under realistic load conditions
- Cross-browser and mobile compatibility

**Performance Tests**: Resource usage and scalability validation
- Concurrent game load testing
- Monte Carlo agent performance benchmarking
- WebSocket connection scalability
- Memory leak detection and prevention

### Expected Performance Targets

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| MC Agent Response | < 5 seconds | Action completion time |
| WebSocket Latency | < 100ms | Message round-trip time |
| Concurrent Games | 3 stable | Load testing validation |
| Memory Usage | < 1GB total | System monitoring |
| Game Completion Rate | > 95% | Success ratio tracking |

## Success Criteria

### Functional Requirements

**Complete User Scenarios**: All specified user workflows implemented and tested
**AI Integration**: Full access to existing Monte Carlo agent capabilities without compromise
**Real-Time Performance**: Responsive multiplayer experience with sub-second action processing
**Reliability**: Graceful error handling and automatic recovery from connection issues

### Technical Requirements

**Local Hosting**: Successful deployment on developer laptop with external accessibility
**Performance**: Monte Carlo agent operating at full capacity (n=300+ simulations)
**Scalability**: Support for multiple concurrent games with configurable resource limits
**Maintainability**: Clean, testable code following established project TDD principles

### Quality Assurance

**Test Coverage**: > 95% code coverage for new components following existing TDD patterns
**Documentation**: Comprehensive setup guides and API documentation
**User Experience**: Intuitive interface with clear feedback and error messaging
**Security**: Robust input validation and secure network communication

## Future Extension Opportunities

### Advanced Features

**Spectator Mode**: Allow observers to watch ongoing games
**Tournament System**: Multi-round competitions with bracket management
**Statistics Tracking**: Player performance analytics and historical data
**Mobile Application**: Progressive Web App (PWA) for native mobile experience

### AI Enhancements

**Difficulty Customization**: Fine-grained AI parameter tuning for personalized challenge levels
**Learning Opponents**: Adaptive AI that learns from player behavior patterns
**Advanced Algorithms**: Integration of ISMCTS and neural network-based agents
**Multi-Agent Research**: Platform for reinforcement learning experimentation

### Infrastructure Scaling

**Cloud Migration Path**: Smooth transition to cloud hosting when user base grows
**Database Scaling**: Migration from SQLite to PostgreSQL for larger deployments
**Containerization**: Docker deployment for simplified installation and distribution
**API Development**: REST API for third-party integrations and mobile applications

## Conclusion

This specification provides a comprehensive roadmap for developing a fully-featured Perudo web frontend that maximizes the capabilities of the existing sophisticated AI agents through local hosting. The approach ensures zero computational compromises while delivering a modern, responsive multiplayer experience.

The implementation leverages the project's strong TDD foundation and modular architecture to create a maintainable, extensible platform that serves as an excellent showcase for the advanced Monte Carlo agent capabilities while providing engaging gameplay for users.

Key advantages of this approach include immediate deployment capability, full AI feature access, zero ongoing costs, and a scalable foundation for future enhancements as the project evolves.