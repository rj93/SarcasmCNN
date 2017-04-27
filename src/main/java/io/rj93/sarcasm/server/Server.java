package io.rj93.sarcasm.server;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.glassfish.grizzly.http.server.HttpServer;
import org.glassfish.jersey.grizzly2.httpserver.GrizzlyHttpServerFactory;
import org.glassfish.jersey.server.ResourceConfig;

import java.io.IOException;
import java.net.URI;

/**
 * Main class.
 *
 */
public class Server {
	
	private static final Logger logger = LogManager.getLogger(Server.class);
	
    private final String BASE_URI;
	private final int port;
    private HttpServer server;
    private final ResourceConfig rc = new ResourceConfig()
    		.packages("io.rj93.sarcasm.server")
			.register(CORSFilter.class);
    
    public Server(){
    	this(8080);
    }
    
    public Server(int port){
    	this.port = port;
    	BASE_URI = "http://0.0.0.0:" + port + "/";
    }

    public void start() {
    	logger.info("Starting server...");
        server = GrizzlyHttpServerFactory.createHttpServer(URI.create(BASE_URI), rc);
        logger.info("Server started at: " + BASE_URI);
    }
    
    public void stop(){
    	logger.info("Stopping server...");
    	server.shutdown();	
    	logger.info("Server stopped.");
    }
    
    public static void main(String[] args) throws IOException {
        Server server = new Server();
        server.start();
        System.in.read();
        server.stop();
    }
}

