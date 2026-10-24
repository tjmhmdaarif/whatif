/* Replace the header's "Open localhost" button with a link back to the landing page.

   The real interface opens http://127.0.0.1:8765, which only exists if the
   visitor is running the tool themselves. Registered after app.js so it runs
   after wire() has attached its handler; swapping the node discards that handler.
*/
document.addEventListener("DOMContentLoaded", function () {
  var button = document.getElementById("btnServe");
  if (!button) return;
  var link = document.createElement("a");
  link.id = "btnServe";
  link.className = "btn btn-ghost";
  link.href = "../";
  link.textContent = "Landing page";
  button.replaceWith(link);
});
