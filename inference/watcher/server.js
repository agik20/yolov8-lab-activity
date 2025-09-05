const WebSocket = require('ws');
const mysql = require('mysql2/promise');
const chokidar = require('chokidar');
const fs = require('fs');
const path = require('path');

// ========== CONFIGURATION ==========
const config = {
    PORT: process.env.PORT || 3001,
    WATCH_FOLDER: path.normalize('inference/output'),
    TIME_TOLERANCE_MS: 3000,
    DB_CONFIG: {
        host: 'localhost',
        user: 'hpw1',
        password: 'hpw123',
        database: 'capstone',
        waitForConnections: true,
        connectionLimit: 10,
        queueLimit: 0
    },
    ALLOWED_EXTENSIONS: ['.jpg', '.jpeg', '.png'],
    FOLDER_MAPPINGS: {
        'pose': 'pose_images',
        'object': 'object_images',
        'room-view': 'room_view'
    }
};

// ========== DATABASE SERVICE ==========
class DatabaseService {
    constructor() {
        this.pool = mysql.createPool(config.DB_CONFIG);
    }

    async insertImageRecord(tableName, filename, folder) {
        const connection = await this.pool.getConnection();
        try {
            // Explicitly specify columns to avoid id field issues
            const [res] = await connection.query(
                `INSERT INTO ${tableName} (filename, folder) VALUES (?, ?)`,
                [filename, folder]
            );
            return res.insertId;
        } catch (err) {
            console.error(`Database error in ${tableName}:`, err);
            throw err; // Re-throw to handle in the calling function
        } finally {
            connection.release();
        }
    }

    async close() {
        await this.pool.end();
    }
}

// ========== FILE SERVICE ==========
class FileService {
    static isValidFilePath(filePath, baseFolder) {
        const relative = path.relative(baseFolder, filePath);
        return relative && !relative.startsWith('..') && !path.isAbsolute(relative);
    }

    static getFolderInfo(filePath) {
        const parentFolder = path.basename(path.dirname(filePath));
        let grandParentFolder = path.basename(path.dirname(path.dirname(filePath)));

        // Special case for room-view
        if (parentFolder === 'room-view') {
            grandParentFolder = 'room-view';
        }

        return { parentFolder, grandParentFolder };
    }
}

// ========== WEBSOCKET SERVICE ==========
class WebSocketService {
    constructor(port) {
        this.server = new WebSocket.Server({ port });
        this.clients = new Set();

        this.server.on('connection', (ws) => {
            this.clients.add(ws);
            ws.on('close', () => this.clients.delete(ws));
        });
    }

    broadcast(payload) {
        const message = JSON.stringify(payload);
        this.clients.forEach(client => {
            if (client.readyState === WebSocket.OPEN) {
                client.send(message);
            }
        });
    }

    close() {
        return new Promise((resolve) => {
            this.server.close(() => resolve());
        });
    }
}

// ========== FILE PROCESSING PIPELINE ==========
class FileProcessingPipeline {
    constructor() {
        this.dbService = new DatabaseService();
        this.wsService = new WebSocketService(config.PORT);
        this.watcher = chokidar.watch(config.WATCH_FOLDER, {
            ignoreInitial: true,
            persistent: true,
            awaitWriteFinish: {
                stabilityThreshold: 2000,
                pollInterval: 100
            },
            depth: 5
        });
    }

    async initialize() {
        this.setupFileWatcher();
        this.setupShutdownHandlers();
        console.log(`WebSocket Server berjalan di port ${config.PORT}`);
        console.log(`Memantau folder: ${config.WATCH_FOLDER}`);
    }

    setupFileWatcher() {
        this.watcher.on('add', async (filePath) => {
            try {
                await this.processNewFile(filePath);
            } catch (err) {
                console.error('Gagal memproses file baru:', err);
            }
        });
    }

    async processNewFile(filePath) {
        // Validate file
        const ext = path.extname(filePath).toLowerCase();
        if (!config.ALLOWED_EXTENSIONS.includes(ext)) return;

        if (!FileService.isValidFilePath(filePath, config.WATCH_FOLDER)) {
            console.warn(`Invalid file path: ${filePath}`);
            return;
        }

        // Extract file information
        const filename = path.basename(filePath);
        const { parentFolder, grandParentFolder } = FileService.getFolderInfo(filePath);

        console.log(`File baru terdeteksi: ${filename}`);
        console.log(`parentFolder: ${parentFolder}`);
        console.log(`grandParentFolder: ${grandParentFolder}`);

        // Prepare WebSocket payload
        const payload = {
            type: 'new-image',
            folder: `${grandParentFolder}/${parentFolder}`,
            filename: filename,
            timestamp: new Date().toISOString()
        };

        // Broadcast to clients
        this.wsService.broadcast(payload);

        // Save to database
        const tableName = config.FOLDER_MAPPINGS[grandParentFolder];
        if (tableName) {
            console.log(`→ Menyimpan ke tabel ${tableName}...`);
            const insertId = await this.dbService.insertImageRecord(tableName, filename, parentFolder);
            console.log(`✓ ${tableName} inserted:`, insertId);
        } else {
            console.warn(`⚠ Folder tidak dikenali: ${grandParentFolder}`);
        }
    }

    setupShutdownHandlers() {
        const cleanup = async () => {
            console.log('Menutup server...');
            try {
                await this.watcher.close();
                await this.dbService.close();
                await this.wsService.close();
                console.log('Server ditutup dengan bersih');
                process.exit(0);
            } catch (err) {
                console.error('Kesalahan saat menutup:', err);
                process.exit(1);
            }
        };

        process.on('SIGINT', cleanup);
        process.on('SIGTERM', cleanup);
    }
}

// ========== MAIN EXECUTION ==========
(async () => {
    try {
        const pipeline = new FileProcessingPipeline();
        await pipeline.initialize();
    } catch (err) {
        console.error('Gagal memulai server:', err);
        process.exit(1);
    }
})();