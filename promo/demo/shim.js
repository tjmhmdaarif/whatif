/* Answer every API call from the captured payloads in data.js.

   The interface only ever talks to the network through fetch(), so overriding
   that one function turns the real app.js into a static page. Anything not in
   data.js is reported as a 404 rather than sent to a server that does not exist.
*/
(function () {
  var DATA = window.__WHATIF_DEMO__ || {};
  var JSON_HEADERS = { "Content-Type": "application/json; charset=utf-8" };

  function reply(status, payload) {
    return Promise.resolve(
      new Response(JSON.stringify(payload), { status: status, headers: JSON_HEADERS })
    );
  }

  window.fetch = function (input, init) {
    var url =
      typeof input === "string" ? input : (input && input.url) || "";
    var method = (
      (init && init.method) ||
      (input && input.method) ||
      "GET"
    ).toUpperCase();
    var key = method + " " + url.split("?")[0];

    if (Object.prototype.hasOwnProperty.call(DATA, key)) {
      return reply(200, DATA[key]);
    }
    if (method !== "GET") {
      var readKey = "GET " + url.split("?")[0];
      if (Object.prototype.hasOwnProperty.call(DATA, readKey)) {
        return reply(200, DATA[readKey]);
      }
    }
    return reply(404, { error: "Not part of this static demo: " + key, status: 404 });
  };
})();
