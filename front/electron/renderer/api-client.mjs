// Desktop uses IPC; the web build talks to the API on the site's own origin.
import {backendContract, resolveAPI, prepareRequest, adaptResponse} from '../../backend-contract.mjs';
export async function request(route, body) {
  const prepared = prepareRequest(route, body);
  if (globalThis.window?.cureDesktop?.request) {
    return adaptResponse(route, await window.cureDesktop.request(prepared.route, prepared.body));
  }
  const {path, method} = resolveAPI(prepared.route, prepared.body);
  const response = await fetch(path, {
    method, credentials: 'same-origin',
    ...(prepared.body === undefined ? {} : {headers: {'Content-Type': 'application/json'}, body: JSON.stringify(prepared.body)}),
    signal: AbortSignal.timeout(backendContract.requestTimeoutMs),
  });
  if (!response.ok) throw new Error((await response.text()).slice(0, 500) || `HTTP ${response.status}`);
  return adaptResponse(route, await response.json());
}
