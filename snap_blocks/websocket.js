/* (prototype) Websocket extension for Snap!
 * ===========================================
 *
 * Testing the possiblities and conceptualizing
 * the use of Websockets in Snap!
 *
 * Note: this script is designed to be used
 * with in Snap!.
 */

function is_instance_skt(fn) {
  return function (socket, ...args) {
    if (!(socket instanceof WebSocket)) {
      throw new TypeError("Argument is not a WebSocket instance");
    }
    return fn(socket, ...args);
  };
}

SnapExtensions.primitives.set(
  'reload(url)',
  function (url,proc) {
    // Helper function to trick snap to reload the extension 
    const index = SnapExtensions.scripts.indexOf(url);
    if (index > -1) {
      SnapExtensions.scripts.splice(index, 1);
    }
  }
);

SnapExtensions.primitives.set(
  'skt_create(url)',
  function (url,proc) {
    try{
      const socket = new WebSocket(url);
      return socket;
    } catch(e) {
      throw e;
    }
  }
);

SnapExtensions.primitives.set(
  'skt_send(socket,data)',
  is_instance_skt(function(socket,data,proc) {
    try{
      socket.send(data);
    } catch(e) {
      throw e;
    }
  })
);

SnapExtensions.primitives.set(
  'skt_close(socket)',
  is_instance_skt(function(socket,proc) {
    socket.close();
  })
);

SnapExtensions.primitives.set(
  'skt_get(socket, property)',
  is_instance_skt(function(socket,property,proc) {
    if (!(property in socket)) {
      throw new ReferenceError(`Property '${property}' does not exist on WebSocket`);
    }
    return socket[property];
  })
);
