import struct
from struct import unpack
import binascii
import socket
import re
import zlib
import base64
import math
import quopri
import json
import base64
from xml.etree import ElementTree as ET

THREADHOLDER_ENTROPY = 6.7
def calculate_entropy(payload):
    """
    데이터의 엔트로피(Entropy)를 계산하는 함수
    - 값이 높을수록 랜덤성이 크며, 암호화된 데이터일 가능성이 높음
    """
    if not payload:
        return 0

    byte_counts = {byte: payload.count(byte) for byte in set(payload)}
    entropy = -sum((count / len(payload)) * math.log2(count / len(payload)) for count in byte_counts.values())
    return entropy

def bts(payload):
    decoded = ""
    for b in payload:
        if 32 <= b <= 126:
        # if 32 <= b <= 126:
            if b == "\x02":
                decoded += "."
            else:
                decoded += chr(b)
        elif b == 0:
            decoded += " "
    decoded = re.sub(r'\s+', ' ', decoded)
    return decoded
def bytes_to_str_32126(payload, protocol="Unknown"):
    entropy = calculate_entropy(payload)
    # encryption_status = "Possibly Encrypted" if entropy > THREADHOLDER_ENTROPY else "Possibly Structured Data"
    decoded = bts(payload)
    encryption_status = "Possibly Encrypted" if entropy > THREADHOLDER_ENTROPY else "Possibly Structured Data"
    if protocol == "Unknown":
        protocol = "Unkown(Possibly Encrypted or Custom Protocol)"
    result = (
        f"Detected Protocol: {protocol}\n"
        f"Entropy: {entropy:.2f}\n"
        f"Encryption Status: {encryption_status}\n"
        f"Signature: {payload[:8].hex()} (first 8 bytes)\n"
        f"Length: {len(payload)} bytes\n"
        f"Readable ASCII Data: {decoded}\n"
    )
    return result

def is_rlp_encoded(payload):
    """
    이더리움 DevP2P RLP 인코딩 여부 검사
    - RLP는 Ethereum DevP2P에서 사용되며, 0x80 이상 바이트 패턴을 포함할 수 있음
    """
    return any(byte >= 0x80 for byte in payload[:10])  # 첫 10바이트만 검사
def is_suspicious_custom_botnet(payload: bytes) -> bool:
    suspicious_patterns = [
        # (start_with, must_contain1, must_contain2)
        (b'\x10.(4v', b'Qxnqn', b'qpSX431'),  # 8083
        (b'?. w', b'giaSmM7', b'OI{22Uvn'),  # 8083
        (b'r.~%+6a', b'UaF?', b'>W8T'),  # 8082
        (b').\\', b'OVVRR4rip9b', b'qon7f'),  # 8082 패턴
        (b'+.G8 !{Sgd@', b'fglHDZ$', b'UA||'),  # 1900
        (b'?.m', b'CJJc', b'pZeM~'),  # 8081
        (b'\n.O', b'OXW_DD', b'Wue'),  # 49245
        (b'F.\x11&', b'WIy9Gj', b'ma~qt'),  # 1027
    ]

    for start_seq, keyword1, keyword2 in suspicious_patterns:
        if payload.startswith(start_seq) and keyword1 in payload and keyword2 in payload:
            return True

    return False


def decode_payload(port, payload):
    def detect_protocol(payload):
        """
        패킷 데이터를 기반으로 프로토콜을 자동 감지하는 함수
        - 포트 정보 없이도 특정 프로토콜을 감지할 수 있도록 설계됨
        """
        if len(payload) < 4:
            return -1

        header = payload[:5]  # 첫 4바이트를 확인
        entropy = calculate_entropy(payload)
        # MGCP 메시지 감지: 시작이 RQNT, AUEP, CRCX, DLCX 등으로 시작하며 MGCP 0.1 포함
        mgcp_commands = [b'RQNT', b'AUEP', b'CRCX', b'DLCX', b'NTFY', b'RSIP']

        # DNS 패킷 감지 (Transaction ID + Flags)
        if (header[2] & 0b10000000):  # QR bit == 1 → Response
            return 53 # DNS Response
        elif (payload[2:4] in [b'\x01\x00', b'\x00\x00', b'\x00\x10', b'\x00\x01']):  # 다양한 DNS 요청 플래그
            return 53 # DNS Query
        elif b'\x03dns' in payload and b'msftncsi' in payload:
            return 53
        # NetBIOS Name Service (NBNS) 감지 (Transaction ID + Flags)
        elif len(payload) >= 12 and (header[2] == 0x00 and header[3] in [0x00, 0x10]) and struct.unpack("!H", payload[4:6])[0] > 0:
            return 137  # NBNS

        # LLMNR 감지 (Transaction ID + Flags)
        elif (header[2] == 0x00 and header[3] == 0x00):
            return 5355  # LLMNR Query
        elif (header[2] == 0x80 and header[3] == 0x00):
            return 5355  # LLMNR Response

        # DHCP 패킷 감지 (BOOTP 구조 확인)
        elif header[0] == 0x01 and header[1] == 0x01 and header[2] == 0x06 and header[3] == 0x00:
            return 67  # DHCP

        # SMB 감지 (Signature: `\xffSMB`)
        elif header == b'\xffSMB':
            return 139  # SMB
        # TLS/SSL 핸드셰이크 감지 (`\x16\x03\x01` → TLS 1.0, `\x16\x03\x03` → TLS 1.2+)
        elif (header[0] in [0x14, 0x15, 0x16, 0x17] and header[1:3] in [b'\x03\x01', b'\x03\x02', b'\x03\x03',
                                                                     b'\x03\x04']) or any(sig in payload for sig in [b'\x16\x03\x01', b'\x16\x03\x02', b'\x16\x03\x03', b'\x16\x03\x04']):
            if b'\x00\x00' in payload and b'.' in payload:
                try:
                    sni_start = payload.find(b'.') - 10
                    sni = payload[sni_start:sni_start + 40].decode(errors='ignore')
                    if any(domain in sni for domain in ['ad-', 'dsp.', 'sync-', 'tracker', 'sock']):
                        return "TLS (Suspicious SNI)"
                except:
                    pass
            return "TLS"
        elif b'http://ocsp' in payload or b'http://crl' in payload or b'1.2.840.113549.1.1' in payload:
            return "TLS/PKI"
        # IMAPS 감지 (TLS 암호화된 IMAP 트래픽, 포트 993)
        elif header[0] in [0x16, 0x17] and header[1:3] in [b'\x03\x01', b'\x03\x02', b'\x03\x03',
                                                                           b'\x03\x04']:
            return 993  # IMAPS
        elif (b"* OK" in payload and b"IMAP" in payload)or (b"eastex.net modusMail IMAP4S" in payload) or (b"SMTP" in payload and b"mail.smartinternz.com" in payload) or\
            (b"25mc" in payload or b"gSlz!8q1F9" in payload):
            return "IMAPS"  # TLS로 감싼 IMAP 서비스
            # ✅ TLS/PKI 감지 (OCSP, CRL 및 X.509 인증서 관련 데이터)

        # SMTP 감지 (Mail Transfer Traffic)
        elif any(cmd in payload for cmd in [b'220 ', b'250 ', b'354 ', b'221 ', b'500 ', b'550 ', b'RCPT TO:', b'DATA', b'smtp', b"SMTP"]) or payload.startswith(b'421') :
            return 25  # SMTP
        elif b'"jsonrpc"' in payload and b'"method"' in payload and b'"params"' in payload:
            return "XMRig_Mining"
        # mDNS 감지 (Transaction ID가 0x0000)
        elif struct.unpack("!H", header[:2])[0] == 0x0000:
            return 5353  # mDNS

        # DNS-SRV 감지 (`_ldap._tcp.` 또는 `_kerberos._tcp.` 같은 서비스 탐색 패턴 포함)
        elif b'_ldap._tcp' in payload or b'_kerberos._tcp' in payload:
            return 389
        # LDAP 감지 (Active Directory 관련 OID 패턴 포함)
        elif b'1.2.840.113556' in payload or b'1.3.6.1.4.1.1466' in payload:
            return "LDAP"
        # elif header[:3] in [b'\x05\x00\x0b', b'\x05\x00\x0c']:
        elif header[:2] in [b'\x05\x00', b'\x04\x00']:
            return 135  # DCE/RPC Bind Request
            # ✅ RPCBind 감지 (Sun RPC, Port 111)
        # Kerberos 감지 (Port 88, ASN.1 구조 및 LDAP 연계 확인)
        elif header[:1] in [b'\xa0', b'\x60', b'\x6e', b'\x03'] and (
                b'1.2.840.113554.1.2.2' in payload or b'krb' in payload or b'\xa1' in payload):
            return 88  # Kerberos
        elif header == b'HTTP' or b"HTTP" in payload or b"<html>" in payload:#HTTP or HTTPS
            return 80
        elif b"xml" in header:
            return "XML"
        # Ethereum DevP2P 감지 (포트 30301)
        elif any(x in payload for x in [b'hello', b'Ethereum', b'devp2p', b'status', b'findnode', b'ping']):
            return "Ethereum DevP2P"  # 이더리움 P2P 프로토콜
        elif b"sip" in payload or b"SDP" in payload:
            return 5060
        elif b"Revenge-RAT" in payload or b'JAB' in payload[:10]:#RAT
            return "Revenge-RAT"
        elif any(payload.find(keyword) != -1 for keyword in [b'NICK ', b'JOIN ', b'PRIVMSG ', b'PING ', b'QUIT ', b'NOTICE ', b'MODE ']):
            return 6667
        elif (b"ftp" in payload or b"FPT" in payload) and (payload.startswith(b"dr") or payload.startswith(b"-r") or b"->" in payload):
            return 20  # FTP
        elif b"get_peers" in payload or b"info_hash" in payload or b"BitTorrent" in payload \
                or b'find_node' in payload or payload.startswith(b'A\x00\x18') :
            return "BitTorrent-DHT"
        elif b"IANA WHOIS server" in payload:#WHOIS
            return 43
        elif (header[:2] == b'\x00\x00' and len(payload) > 10) or payload[:2] == b'\x00\x00':
            return "NetBIOS"
        elif any(x in payload for x in [b'gPa', b'Kn,', b'Pa)', b'As0d']):#BOT
            return "BOT"
        elif b'WIN_' in payload:
            return "InfoStealer"
        elif b"rundll32" in payload:
            return "Windows-DLL-Exec"
        elif b'RFB 003.008' in payload or b'RFB' in payload:
            return "VNC"
        elif b'cmd /V' in payload or b'cmd' in payload:
            return "Windows-CMD-Obfuscated"
        elif payload.startswith(b'\xff\x11\x03\xe1'):
            return "Suspicious-Custom-Protocol"
        elif payload.startswith(b"ew"):
            return "Exfiltration-Base64-JSON"
        elif payload[0] == 0x10 and b'MQTT' in payload:
            return 1883 #MQTT
        elif b'.well-known' in payload and b'core' in payload:
            return 5683
        elif b'MGLNDD' in payload:
            return 873
        elif b'DVKT' in payload:
            return 8995 #DVKT
        elif (b"powershell" in payload or b"PowerShell" in payload) or (b'PS C:\\' in payload or b'PS c:\\' in payload):
            return "CUSTOM_POWERSHELL_C2"
        if any(payload.startswith(cmd) for cmd in mgcp_commands) and b'MGCP' in payload:
            return "MGCP"
        elif b'\x01\xbb' in payload or b'\x00\x50' in payload:
            return "custom_domain_tunnel"
        elif b'&&&' in payload and payload.count(b'&&&') >= 2:
            return "CustomC2-Base64"
        elif b'CONNECTIONLESS_TDS' in payload:
            return "TDS"
        elif b'EPSONP' in payload or b'BJNP' in payload or b'MFNP' in payload or b'CANON' in payload:
            return "PRINTER"
        elif header[:2] == b'\x00\x00':
                return -2  # Unknown Protocol
        elif is_suspicious_custom_botnet(payload):
            return "Suspicious-Custom-Botnet"
        # 난독화된 봇넷 명령 가능성 있음
        # 랜덤성이 높은 데이터 감지 (암호화 가능성)
        if entropy > THREADHOLDER_ENTROPY:  # 일반적인 암호화 데이터의 엔트로피 값 (8.0에 가까움) 7.5 > 7.1 > 6.7
            return "PEncrypted"
        return -1  # Unknown Protocol
    """
    네트워크 패킷 페이로드를 사람이 읽을 수 있도록 디코딩하는 함수
    """
    ports = [80, 8080, 443, 53, 20, 21, 25, 587, 465, 143, 110, 67, 68, 123, 5355, 139, 993, 445, 161, 162, 137, 138,
             5353,389, 636, 111, 135, 88, 30301, 30302, 30303, 30304, 30305, 5060,
             6667, 1900, 43, 5500, 1883, 3702, 6881, 500, 5683, 873, 8995, -2]

    if port not in ports:
        port = detect_protocol(payload)
    ## port -1 is known protocol
    decoded = ""
    if len(payload) == 0:
        return "", port
    try:
        if port in [80, 8080, 443]:  # HTTP / HTTPS
            decoded = decode_http(payload)
        elif port == 53:  # DNS
            decoded = decode_dns(payload)
        elif port in [20, 21]:  # FTP
            decoded = decode_ftp(payload)
        elif port in [25]:  # SMTP
            decoded = decode_smtp(payload)
        elif port in [587, 465]: #SMTP response
            decoded = decode_smtp_response(payload)
        elif port in [143, 110]:  # IMAP / POP3
            decoded = decode_email(payload)
        elif port in [67, 68]:  # DHCP
            decoded = decode_dhcp(payload)
        elif port == 123:  # NTP
            decoded = decode_ntp(payload)
        elif port == 5355:  # LLMNR
            decoded = decode_llmnr(payload)
        elif port in [139, 445]:  # SMB (NetBIOS)
            decoded = decode_smb(payload)
        elif port in [161, 162]:  # SNMP
            decoded = decode_snmp(payload)
        elif port in [137, 138]:  # NetBIOS
            decoded = decode_nbns(payload)
        elif port == "TLS":
            decoded = decode_tls(payload)
        elif port == "TLS/PKI": #
            decoded = decode_tls_or_pki(payload)
        elif port == "TLS (Suspicious SNI)":
            decoded = decode_tls_session(payload)
        elif port == 5353:# mDNS
            decoded = decode_mdns(payload)
        elif port in [389, 636]: #389 (LDAP), 636 (LDAPS)
            decoded = decode_dns_srv(payload)
        elif port == "LDAP":
            decoded = decode_ldap(payload)
        elif port in [111, 135]: #RPC
            decoded = decode_dcerpc(payload)
        elif port == "PEncrypted":
            decoded = decoded_PEncrypted(payload)
        elif port == 88: #Kerberos
            decoded = decode_kerberos(payload)
        elif port == 993:
            decoded = decode_tls_application_data(payload)
        elif port == "XMRig_Mining":
            decoded = decode_xmrig_mining_data(payload)
        elif port in [30301, 30302, 30303, 30304, 30305]: #Ethereum DevP2P
            decoded = decode_ethereum_devp2p(payload)
        elif port == 5060: #SIP
            decoded = decode_sip(payload)
        elif port == [6667]:#IRC
            decoded = decode_irc(payload)
        elif port in ["BitTorrent-DHT", 6881]:#BitTorrent DHT
            decoded = decode_bittorrent_dht(payload)
        elif port in [43]:
            decoded = decode_whois(payload, port)
        elif port in ['BOT']:
            decoded = decode_bot(payload, port)
        elif port in ["IMAPS"]:
            decoded = decode_custom_imap_bot(payload, port)
        elif port in ['Revenge-RAT']:
            decoded = decode_revenge_rat(payload)
        elif port in ['InfoStealer']:
            decoded = decode_infostealer(payload, port)
        elif port in ['Exfiltration-Base64-JSON']:
            decoded = base64_decode_if_possible(payload)
        elif port in [1883]:
            decoded = decode_mqtt(payload)
        elif port in ["VNC"]:
            decoded = decode_vnc(payload)
        elif port in ["RDP", 3702]:
            decoded = decode_ws_discovery(payload)
        elif port in [500]:
            decoded = decode_ike(payload)
        elif port in [873]:
            decoded = decode_rsync(payload)
        elif port in ['custom_domain_tunnel']:
            decoded = decode_custom_domain_tunnel(payload)
        elif port in ['CustomC2-Base64']:
            decoded = decode_custom_base64_chunks(payload)
        elif port in ['MGCP']:
            decoded = decode_mgcp(payload)
        elif port in ['TDS']:
            decoded = decode_tds_connectionless(payload)
        elif port in ['PRINTER']:
            decoded = decode_printer(payload)
        elif port == -2: #Null payload
            decoded = detect_null_bytes(payload)
        elif port == -1:
            decoded = bytes_to_str_32126(payload)
        else:
            return bytes_to_str_32126(payload, port), port # 기본적으로 HEX로 반환

        if decoded == -1:
            decoded = bytes_to_str_32126(payload)

        return decoded, port
    except Exception as e:
        return f"Decoding Error: {str(e)}"

# 1. HTTP / HTTPS 디코딩
def decode_http(data):
    return bytes_to_str_32126(data, "HTTP(S)")


# 2. DNS 디코딩
def decode_dns(data):
    try:
        transaction_id = struct.unpack('!H', data[:2])[0]
        flags = struct.unpack('!H', data[2:4])[0]
        qdcount = struct.unpack('!H', data[4:6])[0]
        ancount = struct.unpack('!H', data[6:8])[0]
        decoded = bts(data[9:])
        return f"Detected Protocol: DNS Query - ID: {transaction_id}\nFlags: {flags}, Questions: {qdcount}, Answers: {ancount} Readable ASCII Data: {decoded}"
    except:
        return bytes_to_str_32126(data, "DNS")

# 3. FTP 디코딩
def decode_ftp(payload):
    """
    FTP LIST 명령의 응답을 사람이 보기 좋게 파싱하는 함수
    """
    try:
        lines = payload.decode("utf-8", errors="ignore").strip().splitlines()
        parsed = ["Detected Protocol: FTP LIST Response"]

        for line in lines:
            parts = line.split()
            if len(parts) >= 9:
                file_type = (
                    "Directory" if parts[0].startswith("d") else
                    "Link" if parts[0].startswith("l") else
                    "File"
                )
                size = parts[4]
                date = " ".join(parts[5:8])
                name = " ".join(parts[8:])
                parsed.append(f"{file_type:9} | {size:>6} bytes | {date} | {name}")
            else:
                parsed.append(f"Unparsed: {line}")
        parsed.append(f"End of FTP LIST Response")
        parsed.append(f"Readable ASCII Data: {bts(payload)}")
        return "\n".join(parsed)

    except Exception as e:
        return f"Decoding Error: {str(e)}"

# 4. SMTP 디코딩
def decode_smtp(payload):
    """
    SMTP 이메일 트래픽을 분석하고, 사람이 읽을 수 있도록 디코딩하는 함수
    """
    try:
        # ✅ SMTP 응답 코드 확인
        smtp_responses = re.findall(rb"(\d{3} .+)", payload)
        smtp_response_text = "\n".join([resp.decode(errors="ignore") for resp in smtp_responses])

        # ✅ 이메일 헤더 추출
        email_from = re.search(rb"From:\s?(.+)", payload)
        email_to = re.search(rb"To:\s?(.+)", payload)
        email_subject = re.search(rb"Subject:\s?(.+)", payload)

        from_text = email_from.group(1).decode(errors="ignore") if email_from else "N/A"
        to_text = email_to.group(1).decode(errors="ignore") if email_to else "N/A"
        subject_text = email_subject.group(1).decode(errors="ignore") if email_subject else "N/A"

        # ✅ 피싱 링크 확인
        phishing_links = re.findall(rb"http[s]?://[^\s]+", payload)
        phishing_links_text = "\n".join([link.decode(errors="ignore") for link in phishing_links])

        # ✅ Base64 및 quoted-printable 데이터 디코딩
        # decoded_body = quopri.decodestring(payload).decode(errors="ignore")
        decoded_body = bts(payload)

        result = (
            f"Detected Protocol: SMTP (Possibly Phishing Email)\n"
            f"SMTP Responses:{smtp_response_text}\n"
            f"From: {from_text}\n"
            f"To: {to_text}\n"
            f"Subject: {subject_text}\n"
            f"Potential Phishing Links:{phishing_links_text}\n"
            f"Readable ASCII Data:{decoded_body}"
        )

        return result

    except Exception as e:
        return f"Decoding Error: {str(e)}"

def decode_smtp_response(payload):
    """
    SMTP 응답 메시지를 디코딩하여 상태 코드 및 주요 메시지를 분석하는 함수
    """
    try:
        # 1. ASCII 디코딩
        text = payload.decode("utf-8", errors="ignore").strip()

        # 2. 상태 코드 추출 (예: 220, 250, 421 등)
        match = re.match(r"^(\d{3})", text)
        status_code = match.group(1) if match else "Unknown"

        # 3. 상태 요약
        status_dict = {
            "220": "Service ready",
            "250": "Requested mail action okay",
            "354": "Start mail input",
            "421": "Service not available / Blocked",
            "450": "Requested action not taken (mailbox busy)",
            "451": "Local error",
            "452": "Insufficient system storage",
            "500": "Syntax error / command unrecognized",
            "550": "Mailbox unavailable / rejected",
            "552": "Storage exceeded",
            "554": "Transaction failed (often used for spam)"
        }
        status_message = status_dict.get(status_code, "Unknown status")

        # 4. 스팸 차단 감지
        is_spam_blocked = any(keyword in text.lower() for keyword in [
            "spam", "blocked", "blacklist", "rejetee", "rejected", "client host"
        ])

        # 5. 차단 IP 추출 (옵션)
        ip_match = re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text)
        blocked_ip = ip_match.group(0) if ip_match else "N/A"

        # 6. 정리된 결과
        result = (
            f"Detected Protocol: SMTP Response Analysis\n"
            f"Status Code: {status_code} ({status_message})\n"
            f"Blocked IP: {blocked_ip}\n"
            f"Spam/Block Detected: {'Yes' if is_spam_blocked else 'No'}\n"
            f"Readable ASCII Data:\n{text}"
        )
        return result

    except Exception as e:
        return f"Decoding Error: {str(e)}"

# 5. 이메일 프로토콜 (IMAP/POP3)
def decode_email(data):
    return data.decode('utf-8', errors='ignore')

# 6. DHCP 디코딩
def decode_dhcp(data):
    try:
        op = data[0]
        htype = data[1]
        hlen = data[2]
        hops = data[3]
        xid = binascii.hexlify(data[4:8]).decode()
        secs = struct.unpack('!H', data[8:10])[0]
        flags = struct.unpack('!H', data[10:12])[0]
        try:
            decode = bts(data[13:])
        except:
            decode = ""
        return f"Detected Protocol: DHCP\n Op: {op}, HType: {htype}, HLen: {hlen}, XID: {xid}, Secs: {secs}, Flags: {flags}\n " \
               f"Readable ASCII Data: {decode}"
    except:
        return bytes_to_str_32126(data, "DHCP")

# 7. NTP 디코딩
def decode_ntp(data):
    try:
        li_vn_mode = data[0]
        stratum = data[1]
        poll = data[2]
        precision = data[3]
        decoded = bts(data[4:])
        result = (
            f"Detected Protocol: LI_VN_MODE({li_vn_mode})\n"
            f"Stratum ID:{stratum}\n"
            f"Poll: {poll}\n"
            f"Precision: {precision}\n"
            f"Readable ASCII Data:{decoded}"
        )
        return result
    except:
        return bytes_to_str_32126(data, "NTP (Unknown)")

# 8. LLMNR 디코딩
def decode_llmnr(data):
    try:
        transaction_id = struct.unpack('!H', data[:2])[0]
        flags = struct.unpack('!H', data[2:4])[0]
        qdcount = struct.unpack('!H', data[4:6])[0]
        decoded = bts(data[7:])
        result = (
            f"Detected Protocol: LLMNR\n"
            f"Query ID:{transaction_id}\n"
            f"Flags: {flags}\n"
            f"Questions: {qdcount}\n"
            f"Readable ASCII Data:{decoded}"
        )
        return result
    except:
        return bytes_to_str_32126(data, "LLMNR(Unknown)")

# 9. SMB (NetBIOS) 디코딩
def decode_smb(data):
    try:
        decoded = bts(data)
        results = (
            f"Detected Protocol: SMB (Server Message Block)\n"
            f"Decoded SMB Data: {binascii.hexlify(data).decode()}\n"
            f"Readable ASCII Data: {decoded}"

        )
        return results
    except:
        return bytes_to_str_32126(data, "SMB")

# 10. SNMP 디코딩
def decode_snmp(data):
    try:
        results = (
            f"Detected Protocol: SNMP\n"
            f"SNMP SMB Data: {binascii.hexlify(data).decode()}\n"
            f"Readable ASCII Data: {bts(data)}"
        )
        return results
    except:
        return bytes_to_str_32126(data, "SNMP (Unknown)")

# 11. nbns 디코딩
def decode_nbns(payload):
    """
    NetBIOS Name Service (NBNS) 패킷 디코딩
    """
    try:
        # Transaction ID (2바이트)
        transaction_id = struct.unpack('!H', payload[:2])[0]

        # Flags (2바이트)
        flags = struct.unpack('!H', payload[2:4])[0]

        # Question Count (2바이트)
        qdcount = struct.unpack('!H', payload[4:6])[0]

        # Answer Count (2바이트)
        ancount = struct.unpack('!H', payload[6:8])[0]

        # Authority Count (2바이트)
        nscount = struct.unpack('!H', payload[8:10])[0]

        # Additional Count (2바이트)
        arcount = struct.unpack('!H', payload[10:12])[0]

        # NetBIOS Name (16바이트 인코딩)
        nb_name_raw = payload[12:44]
        nb_name = decode_netbios_name(nb_name_raw)

        # Query Type (2바이트) & Query Class (2바이트)
        if payload[44:48] == b'':
            query_type = None
            query_class = None
        else:
            try:
                query_type, query_class = struct.unpack('!HH', payload[44:48])
            except struct.error:
                query_type, query_class = None, None

        result = (
            f"Detected Protocol: NetBIOS Name Service (NBNS) Packet\n"
            f"Transaction ID: {transaction_id}\n"
            f"Flags: {flags}\n"
            f"Questions: {qdcount}, Answers: {ancount}, Authority: {nscount}, Additional: {arcount}\n"
            f"Queried Name: {nb_name}\n"
            f"Query Type: {query_type}, Query Class: {query_class}"
        )
        return result


    except Exception as e:
        return bytes_to_str_32126(payload, "not NBSN Packet Unkown")

def decode_netbios_name(data):
    """
    NetBIOS 이름 디코딩 (16바이트 NetBIOS-encoded name)
    """
    return bytes_to_str_32126(data, "NetBIOS Name")

def decode_tls(data):
    """
    TLS 핸드셰이크 패킷을 디코딩하는 함수
    """
    try:
        if len(data) < 5:
            return bytes_to_str_32126(data, "TLS Packet")

        # Content Type (1 byte)
        content_type = data[0]
        content_type_dict = {0x16: "Handshake", 0x15: "Alert", 0x17: "Application Data"}
        content_type_str = content_type_dict.get(content_type, f"Unknown (0x{content_type:02x})")

        # TLS Version (2 bytes)
        version_major, version_minor = struct.unpack("!BB", data[1:3])
        tls_version = f"TLS {version_major}.{version_minor}"

        # Length (2 bytes)
        length = struct.unpack("!H", data[3:5])[0]

        result = (
            f"Detected Protocol: TLS Packet\n"
            f"Content Type: {content_type_str}\n"
            f"TLS Version: {tls_version}\n"
            f"Payload Length: {length} bytes"
        )

        # Handshake Message Parsing (if Handshake)
        if content_type == 0x16 and len(data) > 9:
            handshake_type = data[5]
            handshake_dict = {0x01: "ClientHello", 0x02: "ServerHello", 0x0b: "Certificate"}
            handshake_str = handshake_dict.get(handshake_type, f"Unknown Handshake (0x{handshake_type:02x})")
            result += f"\nHandshake Type: {handshake_str}"

        return result

    except Exception as e:
        return f"Decoding Error: {str(e)}"


def decode_mdns(payload):
    """
    Multicast DNS (mDNS) 패킷을 디코딩하는 함수
    """
    try:
        if len(payload) < 12:
            return "Invalid mDNS Packet"

        # Transaction ID (mDNS는 항상 0x0000)
        transaction_id = struct.unpack("!H", payload[:2])[0]

        # Flags
        flags = struct.unpack("!H", payload[2:4])[0]

        # Questions, Answers, Authority, Additional Counts
        qdcount, ancount, nscount, arcount = struct.unpack("!HHHH", payload[4:12])

        # Query Name (가변 길이)
        offset = 12
        queried_name = []
        try:
            while payload[offset] != 0:
                length = payload[offset]
                offset += 1
                queried_name.append(payload[offset:offset + length].decode(errors="ignore"))
                offset += length
            # Query Type & Query Class
            query_type, query_class = struct.unpack("!HH", payload[offset:offset + 4])

        except IndexError:
            query_type, query_class = None, None
            pass
        except struct.error:
            query_type, query_class = None, None
            pass
        queried_domain = ".".join(queried_name)
        queried_domain = bts(bytes(queried_domain, "utf-8"))
        try:
            decoded = bts(payload[13:])
        except:
            decoded = ""
        result = (
            f"Detected Protocol: mDNS Query Packet\n"
            f"Transaction ID: {transaction_id} (Always 0x0000 for mDNS)\n"
            f"Flags: {flags}\n"
            f"Questions: {qdcount}, Answers: {ancount}, Authority: {nscount}, Additional: {arcount}\n"
            f"Queried Domain: {queried_domain}\n"
            f"Query Type: {query_type}, Query Class: {query_class}"
            f"Readable ASCII Data: {decoded}"
        )

        return result

    except Exception as e:
        return f"Decoding Error: {str(e)}"


## LDAP/DNS-SRV
def decode_dns_srv(payload):
    """
    DNS-SRV (서비스 검색용 DNS 패킷) 디코딩 함수
    """
    try:
        # Transaction ID & Flags
        transaction_id = struct.unpack("!H", payload[:2])[0]
        flags = struct.unpack("!H", payload[2:4])[0]

        # Questions, Answer, Authority, Additional
        qdcount, ancount, nscount, arcount = struct.unpack("!HHHH", payload[4:12])

        offset = 12
        queried_name, offset = parse_domain_name(payload, offset)

        # Query Type & Class
        query_type, query_class = struct.unpack("!HH", payload[offset:offset+4])
        offset += 4
        result = (
            f"Detected Protocol: DNS-SRV Response Packet\n"
            f"Transaction ID: {transaction_id}\n"
            f"Flags: {flags}\n"
            f"Questions: {qdcount}, Answers: {ancount}, Authority: {nscount}, Additional: {arcount}\n"
            f"Queried Service: {queried_name}\n"
            f"Queried Data:\n"
            f"Type: {query_type}, Class: {query_class}\n"
        )
        # 응답 데이터 파싱
        if ancount > 0:
            answer_name, offset = parse_domain_name(payload, offset)
            if answer_name is not None:
                answer_type, answer_class, ttl, rdlength = struct.unpack("!HHIH", payload[offset:offset+10])
                offset += 10
                priority, weight, port = struct.unpack("!HHH", payload[offset:offset+6])
                offset += 6
                target, offset = parse_domain_name(payload, offset)

                result+= (
                    f"Answer Name: {answer_name}\n"
                    f"Answer Type: {answer_type}, Class: {answer_class}, TTL: {ttl}, RDLength: {rdlength}\n"
                    f"Priority: {priority}, Weight: {weight}, Port: {port}\n"
                    f"Target: {target}"
                )
            else:
                result += "DNS-SRV Query (No Answer)"
        try:
            decoded = bts(payload[offset:])
        except:
            decoded = ""
        result += f"\nReadable ASCII Data: {decoded}"
        return result
    except Exception as e:
        decoded = f"Decoding Error: {str(e)}\n"
        decoded += bytes_to_str_32126(payload, "DNS-SRV Response Packet")
        return decoded

def parse_domain_name(payload, offset):
    """
    DNS 패킷에서 도메인 이름을 추출하는 함수 (압축된 도메인도 처리)
    """
    labels = []
    jump_occurred = False  # 압축 포인터가 발생했는지 여부 확인
    initial_offset = offset  # 초기 위치 저장
    MAX_JUMP = 10
    jump_count = 0
    while True:
        if offset >= len(payload):  #패킷 길이 초과 확인
            return None, offset  # 도메인 이름을 찾을 수 없음

        length = payload[offset]

        #도메인 압축 처리 (첫 비트가 0b11000000인 경우)
        if length & 0xC0 == 0xC0:
            pointer_offset = ((length & 0x3F) << 8) | payload[offset + 1]
            if pointer_offset >= len(payload):  # 포인터가 유효한 위치인지 확인
                return None, offset + 2
            offset += 2  # 압축된 포인터는 2바이트
            if not jump_occurred:
                initial_offset = offset  # 압축 발생 시, 현재 위치 저장
            offset = pointer_offset  # 새로운 위치로 점프
            jump_occurred = True  # 압축 포인터 처리됨
            jump_count += 1
            if jump_count > MAX_JUMP:
                return None, offset + 2  # 너무 많은 점프 → 무한 루프 방지
            continue

        #도메인 이름 끝 (`\x00`) 도달 시 종료
        if length == 0:
            return ".".join(labels), (initial_offset if jump_occurred else offset + 1)

        #레이블 추가
        label = payload[offset + 1:offset + 1 + length]
        labels.append(label.decode(errors="ignore"))  # ASCII 변환
        offset += length + 1  # 다음 필드로 이동

    return ".".join(labels), offset
def decode_ldap(payload):
    """
    LDAP 패킷을 디코딩하는 함수
    """
    try:
        # OID 값 추출
        oid_pattern = re.compile(r"(\d+\.\d+\.\d+\.\d+\.\d+\.\d+\.\d+)")
        oid_matches = oid_pattern.findall(payload.decode(errors="ignore"))

        # DN (Distinguished Name) 정보 추출
        dn_pattern = re.compile(r"(DC=[a-zA-Z0-9-]+,DC=[a-zA-Z0-9-]+)")
        dn_matches = dn_pattern.findall(payload.decode(errors="ignore"))

        # CN (Common Name) 정보 추출
        cn_pattern = re.compile(r"(CN=[a-zA-Z0-9-]+(?:,CN=[a-zA-Z0-9-]+)*)")
        cn_matches = cn_pattern.findall(payload.decode(errors="ignore"))

        result = (
            f"Detected Protocol: LDAP Query Packet\n"
            f"Detected OIDs: {', '.join(oid_matches)}\n"
            f"Detected DN: {', '.join(dn_matches)}\n"
            f"Detected CN: {', '.join(cn_matches)}"
        )

        return result

    except Exception as e:
        return -1

def decode_dcerpc(payload):
    """
    DCE/RPC (포트 135) 패킷을 디코딩하는 함수
    """
    try:
        if len(payload) < 24:
            return bytes_to_str_32126(payload, "Detect Protocol: DCE/RPC Packet(Incomplete)")

        # DCE/RPC 헤더 파싱
        version, minor_version, packet_type, flags = struct.unpack("!BBBB", payload[:4])
        frag_length = struct.unpack("!H", payload[8:10])[0]
        auth_length = struct.unpack("!H", payload[10:12])[0]
        call_id = struct.unpack("!I", payload[12:16])[0]

        # 바인드 요청인지 확인 (Packet Type 0x0b = Bind Request)
        packet_type_str = {
            0x00: "Request",
            0x02: "Response",
            0x03: "Fault",
            0x04: "Working",
            0x05: "Cancel",
            0x06: "Orphaned",
            0x0b: "Bind Request",
            0x0c: "Bind Ack",
            0x0d: "Bind Nak",
            0x0e: "Alter Context",
            0x0f: "Alter Context Ack"
        }.get(packet_type, f"Unknown (0x{packet_type:02x})")

        # data_representation 필드가 있는지 확인
        data_representation = "N/A"
        if len(payload) >= 8:
            try:
                data_representation = struct.unpack("!I", payload[4:8])[0]
                data_representation = f"0x{data_representation:08x}"
            except struct.error:
                pass  # 필드가 없으면 "N/A" 유지

        # UUID가 포함된 패킷인지 확인
        if len(payload) >= 40:
            uuid = binascii.hexlify(payload[24:40]).decode()
        else:
            uuid = "N/A"

        try:
            decoded = bts(payload[41:])
        except IndexError:
            decoded = ""
        result = (
            f"Detected Protocol: DCE/RPC Packet\n"
            f"Version: {version}.{minor_version}\n"
            f"Packet Type: {packet_type_str}\n"
            f"Flags: 0x{flags:02x}\n"
            f"Data Representation: {data_representation if isinstance(data_representation, str) else f'0x{data_representation:08x}'}\n"
            f"Fragment Length: {frag_length}\n"
            f"Authentication Length: {auth_length}\n"
            f"Call ID: {call_id}\n"
            f"Service UUID: {uuid}"
            f"Readable ASCII Data: {decoded}"
        )

        return result

    except Exception as e:
        return f"Decoding Error: {str(e)}"


def decoded_PEncrypted(data):
    """
    포트 2287 페이로드 분석 함수
    - Base64, 압축, XOR 암호화 여부 검사
    """
    decoded = ""
    decode_type = None
    ascii = bts(data)
    try:
        # Base64 여부 확인
        try:
            decoded_b64 = base64.b64decode(data).decode(errors="ignore")
            decoded = f"Base64 Decoded: {decoded_b64}"
            decode_type = "Base64"
        except:
            pass  # Base64 아님

        # 압축 데이터 여부 확인 (zlib 압축 해제 시도)
        try:
            decompressed = zlib.decompress(data)
            decoded = f"Decompressed Data: {decompressed.decode(errors='ignore')}"
            decode_type = "Zlib"
        except:
            pass  # 압축 아님

        if decoded == "":
            decoded = bts(data)
            decoded_type = "Unknown"


        result = (
            f"Detected Protocol: PEncrypted\n"
            f"decode_type: {decode_type}\n"
            f"decoded: {decoded}\n"
            f"Readable ASCII Data: {ascii}"
        )

        return result
    except Exception as e:
        return bytes_to_str_32126(data, f"PEncrypted Data({e}_")


def decode_kerberos(payload):
    """
    Kerberos (포트 88) 및 LDAP ASN.1 패킷을 디코딩하는 함수
    - ASN.1 구조 기반으로 패킷을 분석하여 사람이 읽을 수 있도록 변환
    """
    try:
        # ✅ ASN.1 구조에서 타임스탬프 패턴 찾기 (YYYYMMDDHHMMSSZ)
        timestamp_match = re.search(rb"(\d{14}Z)", payload)
        timestamp = timestamp_match.group(1).decode() if timestamp_match else "N/A"

        # ✅ 도메인 정보 추출
        domain_pattern = re.compile(rb"([a-zA-Z0-9.-]+)\x00?")
        domain_matches = domain_pattern.findall(payload)
        domain_info = [match.decode(errors="ignore") for match in domain_matches]

        # ✅ HEX 변환 (암호화된 티켓 또는 LDAP 응답 포함 가능)
        hex_data = bytes_to_str_32126(payload, "kerberos")

        # ✅ 패킷 유형 판별 (Kerberos 또는 LDAP)
        if b'1.2.840.113556' in payload or b'ldap' in payload:
            protocol_type = "LDAP"
        elif timestamp != "N/A":  # Kerberos 티켓 요청 또는 응답 패킷
            protocol_type = "Kerberos"
        else:
            protocol_type = "Unknown"

        decoded = bts(payload)
        result = (
            f"Detected Protocol: {protocol_type} Packet\n"
            f"Timestamp: {timestamp}\n"
            f"Domain Info: {', '.join(domain_info)}\n"
            f"Readable ASCII Data: {decoded}\n"
        )

        return result

    except Exception as e:
        return f"Decoding Error: {str(e)}"

def decode_tls_or_pki(data):
    """
    TLS/SSL 인증 데이터 또는 ASN.1 DER 인코딩 패킷을 디코딩하는 함수
    """
    try:
        # ✅ OCSP 및 CRL URL 확인 (TLS 인증서 검증 과정 가능성 탐색)
        ocsp_match = re.search(rb"(http://[a-zA-Z0-9./-]+)", data)
        ocsp_url = ocsp_match.group(1).decode() if ocsp_match else "N/A"

        crl_match = re.search(rb"(http://[a-zA-Z0-9./-]+\.crl)", data)
        crl_url = crl_match.group(1).decode() if crl_match else "N/A"

        # ✅ HEX 변환 (암호화된 서명 또는 키가 포함될 가능성 있음)
        decoded = bytes_to_str_32126(data, "tls")

        # ✅ 랜덤성 분석 → 암호화 가능성 탐색
        encryption_status = "Possibly Encrypted" if len(set(data)) > 100 else "Not Encrypted"

        result = (
            f"Detected Protocol: TLS/PKI Encoded Packet\n"
            f"OCSP URL: {ocsp_url}\n"
            f"CRL URL: {crl_url}\n"
            f"Encryption Status: {encryption_status}\n"
            f"Readable ASCII Data: {decoded}\n"
        )

        return result

    except Exception as e:
        return bytes_to_str_32126(data, f"TLS/PKI Data({e})")


def detect_null_bytes(data):
    """
    NULL 바이트 (`b'\x00\x00'`)가 포함된 패킷을 분석하는 함수
    """
    if data == b'\x00\x00':
        return "Detected: Empty or NULL Padding Data"
    elif data.startswith(b'\x00\x00'):
        return "Detected: Possible Protocol Padding or Transaction ID"
    elif b'\x00\x00' in data:
        return "Detected: NULL Bytes in Data (Check Context)"
    else:
        return "No NULL Bytes Detected"

def decode_tls_application_data(payload):
    """
    TLS Application Data (암호화된 세션 데이터) 분석 함수
    """
    try:
        if len(payload) < 5:
            return "Invalid TLS Packet"

        # TLS 레코드 헤더 분석
        record_type, version_major, version_minor = struct.unpack("!BHH", payload[:5])
        tls_version = f"{version_major}.{version_minor}"

        # TLS 버전 검증
        if tls_version not in ["3.1", "3.2", "3.3", "3.4"]:  # TLS 1.0 ~ TLS 1.3
            return "Invalid or Unsupported TLS Version"

        # Application Data 여부 확인
        if record_type == 0x17:
            data_type = "TLS Application Data (Encrypted)"
        else:
            data_type = "Unknown TLS Record"

        # HEX 변환 (암호화된 트래픽 포함 가능)
        # hex_data = binascii.hexlify(payload).decode(errors="ignore")
        decoded = bts(payload)


        result = (
            f"Detected Protocol: TLS (Encrypted Traffic)\n"
            f"TLS Version: {tls_version}\n"
            f"Record Type: {data_type}\n"
            f"Readable ASCII Data: {decoded}\n"
        )

        return result

    except struct.error:
        return "Decoding Error: Malformed TLS Packet"
    except Exception as e:
        return f"Decoding Error: {str(e)}"

def decode_tls_session(payload):

    try:
        if payload[:3] not in [b'\x16\x03\x01', b'\x16\x03\x03']:
            return "Not a TLS packet"

        # 추정 길이
        total_len = len(payload)

        # SNI 추출 시도
        sni = None
        sni_start = 0
        if b".ad-" in payload:
            try:
                sni_start = payload.find(b".ad-") - 10
                sni = payload[sni_start:sni_start+40].decode(errors="ignore")
            except:
                pass
        try:
            decoded = bts(payload[sni_start+40:])
        except:
            decoded = ""
        result = (
            f"Detected Protocol: TLS Session Detected\n"
            f"Length: {total_len} bytes\n"
            f"SNI: {sni if sni else 'Unknown'}\n"
            f"Contains: Full TLS Handshake + Application Data\n"
            f"Readable ASCII Data: {decoded}"

        )
        return result
    except Exception as e:
        return f"[Decode Error] {e}"


def decode_xmrig_mining_data(payload):
    """
    XMRig 및 Monero 채굴 트래픽을 분석하는 함수
    """
    try:
        # JSON 개별 메시지 분리
        messages = payload.strip().split(b'\n')

        decoded_results = []
        for msg in messages:
            try:
                json_data = json.loads(msg.decode(errors="ignore"))

                # JSON-RPC 기본 정보
                id = json_data.get("id", "id")
                jsonrpc = json_data.get("jsonrpc", "N/A")
                error = json_data.get("error", "null")
                method = json_data.get("method", "Unknown")
                params = json_data.get("params", {})
                algo = params.get("algo", "N/A")

                # XMRig 버전 및 정보
                agent = params.get("agent", "Unknown Agent")
                login = params.get("login", "N/A")
                job_id = params.get("job_id", "N/A")
                blob = params.get("blob", "N/A")
                target = params.get("target", "N/A")

                result = (
                    f"Detected Protocol: XMRig Mining (Monero/Pool Communication)\n"
                    f"id: {id}\n"
                    f"jsonrpc: {jsonrpc}\n"
                    f"error: {error}\n"
                    f"Method: {method}\n"
                    f"Algorithm: {algo}\n"
                    f"Job ID: {job_id}\n"
                    f"Agent: {agent}\n"
                    f"Login: {login}"
                    f"blob: {blob}"
                    f"target: {target}"
                    f"blob: {blob}"
                )

                decoded_results.append(result)
            except json.JSONDecodeError:
                continue  # JSON 디코딩 실패한 경우 무시
        decoded = bts(payload)
        decoded_results.append(decoded)
        return "\n".join(decoded_results)

    except Exception as e:
        return f"Decoding Error: {str(e)}"

def decode_ethereum_devp2p(payload):
    """
    포트 30301의 데이터 분석 함수
    - 이더리움 P2P 프로토콜(DevP2P) 가능성 확인
    - 사람이 읽을 수 있는 ASCII 데이터 추출
    - 암호화 여부 분석
    """
    try:
        # ✅ Entropy 계산
        entropy = calculate_entropy(payload)

        # ✅ 사람이 읽을 수 있는 ASCII 문자열 변환
        readable_data = "".join(chr(b) if 32 <= b <= 126 else "." for b in payload)

        # ✅ HEX 변환 (바이너리 데이터 포함 가능)
        hex_data = binascii.hexlify(payload).decode(errors="ignore")

        # ✅ 이더리움 DevP2P(30301 포트) 가능성 체크
        ethereum_p2p_keywords = [b"discovery", b"Ethereum", b"devp2p", b"hello"]
        is_ethereum_p2p = any(keyword in payload for keyword in ethereum_p2p_keywords)

        # ✅ 암호화 가능성 판별
        encryption_status = "Possibly Encrypted" if entropy > 7.5 else "Possibly Structured Data"

        result = (
            f"Detected Protocol: {'Ethereum DevP2P' if is_ethereum_p2p else 'Unknown (Possibly Custom Protocol)'}\n"
            f"Entropy: {entropy:.2f}\n"
            f"Encryption Status: {encryption_status}\n"
            f"Readable ASCII Data: {readable_data}\n"
        )

        return result

    except Exception as e:
        return f"Decoding Error: {str(e)}"

def decode_sip(payload):
    """
    SIP(Session Initiation Protocol) 패킷을 디코딩하는 함수
    """
    try:
        # ✅ SIP 패킷을 UTF-8로 디코딩 (텍스트 기반)
        sip_message = payload.decode("utf-8", errors="ignore")

        # ✅ 첫 줄 (SIP 요청 또는 응답 라인) 추출
        first_line = sip_message.split("\r\n")[0]

        # ✅ 주요 SIP 헤더 필드 정규식
        sip_headers = {
            "Method": re.search(r"^(INVITE|ACK|OPTIONS|BYE|CANCEL|REGISTER|INFO|PRACK|UPDATE|SUBSCRIBE|NOTIFY|PUBLISH|MESSAGE|REFER) ", first_line),
            "Response": re.search(r"^SIP/2.0 (\d{3}) (.+)", first_line),
            "From": re.search(r"From: (.+)", sip_message),
            "To": re.search(r"To: (.+)", sip_message),
            "Call-ID": re.search(r"Call-ID: (.+)", sip_message),
            "CSeq": re.search(r"CSeq: (.+)", sip_message),
            "Via": re.search(r"Via: (.+)", sip_message),
            "Content-Length": re.search(r"Content-Length: (\d+)", sip_message)
        }

        # ✅ SIP 메시지 유형 확인
        if sip_headers["Method"]:
            sip_type = f"SIP Request {sip_headers['Method'].group(1)}"
        elif sip_headers["Response"]:
            sip_type = f"SIP Response {sip_headers['Response'].group(1)} {sip_headers['Response'].group(2)}"
        else:
            sip_type = "Unknown SIP Message"

        decoded = bts(payload)
        # ✅ SIP 메시지 정리
        result = (
            f"Detected Protocol: SIP\n"
            f"Message Type: {sip_type}\n"
            f"From: {sip_headers['From'].group(1) if sip_headers['From'] else 'N/A'}\n"
            f"To: {sip_headers['To'].group(1) if sip_headers['To'] else 'N/A'}\n"
            f"Call-ID: {sip_headers['Call-ID'].group(1) if sip_headers['Call-ID'] else 'N/A'}\n"
            f"CSeq: {sip_headers['CSeq'].group(1) if sip_headers['CSeq'] else 'N/A'}\n"
            f"Via: {sip_headers['Via'].group(1) if sip_headers['Via'] else 'N/A'}\n"
            f"Content-Length: {sip_headers['Content-Length'].group(1) if sip_headers['Content-Length'] else 'N/A'}\n"
            f"Readable ASCII Data: {decoded}"
        )

        return result

    except Exception as e:
        return f"Decoding Error: {str(e)}"


def decode_irc(payload):
    """
    IRC (Internet Relay Chat) 패킷을 디코딩하는 함수
    """
    try:
        # ✅ IRC 메시지를 UTF-8로 디코딩 (텍스트 기반)
        irc_message = payload.decode("utf-8", errors="ignore")
        decoded = bts(payload)
        # ✅ 주요 IRC 명령어 및 정보 추출
        irc_data = {
            "Nick": re.findall(r"NICK (\S+)", irc_message),
            "Join": re.findall(r"JOIN :(\S+)", irc_message),
            "PrivMsg": re.findall(r"PRIVMSG (\S+) :(.*)", irc_message),
            "Quit": re.findall(r"QUIT :(.*)", irc_message)
        }

        # ✅ PRIVMSG에서 감염된 PC 정보 추출
        system_info = []
        for _, msg in irc_data["PrivMsg"]:
            info_match = re.findall(r"\|!\|Info\|!\|(.+?)\|!\|(.+?)\|!\|(.+?)\|!\|(.+?)\|!\|(.+?)\|!\|(.+?)\|!\|(.+?)\|!\|(.+?)\|!\|(.+?)\|!\|(.+?)\|!\|(.+?)\|!\|(.+?)\|!\|(.+?)\|!\|(.+?)\|!\|(.+?)\|!", msg)
            if info_match:
                system_info.append(info_match[0])

        result = (
            f"Detected Protocol: IRC (Possibly Botnet C2)\n"
            f"Nicknames: {', '.join(irc_data['Nick'])}\n"
            f"Joined Channels: {', '.join(irc_data['Join'])}\n"
            f"Quit Messages: {', '.join(irc_data['Quit'])}\n"
            f"PrivMsg System Info: {system_info}\n"
            f"Readable ASCII Data: {decoded}\n"
        )

        return result

    except Exception as e:
        return f"Decoding Error: {str(e)}"


def decode_bittorrent_dht(payload):
    """
    BitTorrent DHT + Handshake 혼합 트래픽 분석
    """
    try:
        decoded = bts(payload)
        text = payload.decode("latin1", errors="ignore")

        # DHT 정보 추출
        query = re.search(r'1:q\d+:(\w+)', text)
        node_id = re.search(r'2:id20:(.{20})', text, re.DOTALL)
        target = re.search(r'6:target20:(.{20})', text, re.DOTALL)
        nodes = b'nodes' in payload

        # Handshake 여부 확인
        handshake = b'BitTorrent protocol' in payload

        def to_hex(s): return binascii.hexlify(s.encode("latin1")).decode()

        result = f"BitTorrent Traffic Detected\n"

        if query:
            result += f"DHT Query: {query.group(1)}\n"
        if node_id:
            result += f"Node ID: {to_hex(node_id.group(1))}\n"
        if target:
            result += f"Target ID: {to_hex(target.group(1))}\n"
        if nodes:
            result += f"Contains DHT 'nodes' response\n"
        if handshake:
            result += f"Includes BitTorrent Handshake Message\n"
        if decoded:
            result += f"Readable ASCII Data: {decoded}\n"
        return result.strip()

    except Exception as e:
        return f"Decoding Error: {str(e)}"

def decode_whois(payload, port=43):
    try:
        text = payload.decode(errors="ignore")
        lines = text.strip().splitlines()
        result_lines = []

        for line in lines:
            if line.strip().startswith('%'):
                continue  # 주석 제거
            result_lines.append(line.strip())
        decoded = bts(payload)
        result_lines.append(f"Readable ASCII Data: {decoded}")
        return (
            f"WHOIS Response (Port: {port})\n" +
            "\n".join(result_lines) # 너무 길 경우 일부만 출력
        )

    except Exception as e:
        return f"Decoding Error: {str(e)}"


def decode_bot(payload, port=None):
    try:
        ascii_preview = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in payload)
        return (
            f"Custom C2/Bot Protocol Detected (Port {port})\n"
            f"Length: {len(payload)} bytes\n"
            f"ASCII Preview: {ascii_preview[:100]}\n"
            f"Notable Tokens: {[t for t in ['gPa', 'As0d', 'Kn,', 'tkA'] if t.encode() in payload]}"
            f"Readable ASCII Data: {ascii_preview}"
        )
    except Exception as e:
        return f"Decoding Error: {str(e)}"


def decode_custom_imap_bot(payload, port=None):
    try:
        result = []
        decoded = bts(payload)

        if b"* OK" in payload and b"IMAP" in payload:
            result.append("📥 Detected: IMAP greeting or server banner")
        if b"eastex.net modusMail IMAP4S" in payload:
            result.append("📡 Custom IMAPS service: eastex.net modusMail")
        if b"SMTP" in payload:
            result.append("📧 SMTP indicator found")
        if b"25mc" in payload:
            result.append("🔁 Repeated custom marker: 25mc")

        return (
            f"Custom Protocol Analysis (Port {port})\n"
            f"Length: {len(payload)} bytes\n"
            f"Indicators:\n  - " + "\n  - ".join(result) + "\n"
            f"Readable ASCII Data: {decoded}"
        )
    except Exception as e:
        return f"Decoding Error: {str(e)}"


def decode_revenge_rat(payload):
    try:
        # 정규 표현식으로 base64 문자열 추출
        base64_strings = re.findall(rb'[A-Za-z0-9+/=]{20,}', payload)

        decoded_segments = [f"Detected Protocol: RAT (RevengeRAT)"]
        for b64 in base64_strings:
            try:
                decoded = base64.b64decode(b64)
                decoded_segments.append(decoded.decode('utf-16', errors='ignore'))
            except:
                continue

        decoded = bts(payload)
        decoded_segments.append(f"Readable ASCII Data: {decoded}")
        return "\n---\n".join(decoded_segments)
    except Exception as e:
        return f"Decoding Error: {str(e)}"


def decode_infostealer(payload, port="Unknown"):
    try:
        decoded = payload.decode(errors='ignore')
        fields = decoded.split('|')
        labels = [
            "Country", "BotID", "IP", "Username", "OS", "Architecture",
            "Time", "Admin Rights", "Bot Version", "Campaign ID",
            "Resolution", "System Type", "Unknown", "Mutex"
        ]
        result = f"Detected Protocol: Infostealer({port})\n"
        result += "\n".join(f"{k}: {v}" for k, v in zip(labels, fields))
        decoded = bts(payload)
        result += f"\nReadable ASCII Data: {decoded}"
        return result
    except Exception as e:
        return f"Decoding Error: {str(e)}"


def base64_decode_if_possible(payload: bytes) -> str:
    """
    주어진 바이트 스트림(payload)이 Base64로 인코딩된 경우,
    이를 디코딩하여 사람이 읽을 수 있는 문자열로 반환.
    디코딩이 불가능하면 빈 문자열을 반환.

    Args:
        payload (bytes): 원시 페이로드

    Returns:
        str: 디코딩된 문자열 (또는 실패 시 "")
    """
    try:
        # Base64 의심 블럭 찾기
        match = re.search(rb'[A-Za-z0-9+/=]{100,}', payload)
        if not match:
            return ""

        base64_part = match.group()
        cleaned = re.sub(rb'[^A-Za-z0-9+/=]', b'', base64_part)

        usable_len = len(cleaned) - (len(cleaned) % 4)
        cleaned = cleaned[:usable_len]
        cleaned = base64.b64decode(cleaned).decode("utf-8", errors="ignore")
        results = (
            f"Detected Protocol: Unknown(Base64 Encoded Data)\n"
            f"Base64 Decoded Data:{cleaned}\n"
            f"Base64 Encoded Data:{base64_part.decode()}"
            f"Readable ASCII Data: {bts(payload)}"
        )
        return results
    except Exception as e:
        return f"Decoding Error: {str(e)}"

def decode_custom_base64_chunks(payload: bytes) -> str:
    try:
        text = payload.decode(errors="ignore")
        parts = text.split("&&&")
        decoded_parts = []
        for part in parts:
            try:
                decoded = base64.b64decode(part).decode(errors="ignore")
                decoded_parts.append(decoded)
            except Exception:
                decoded_parts.append(f"(invalid base64): {part}")
        return (
            f"Detected Protocol: Unknown(Based Base64)\n"
            "Custom Base64 Chunked Protocol Detected\n"
            + "\n".join(f"Chunk {i+1}: {d}" for i, d in enumerate(decoded_parts))
            + f"\nReadable ASCII Data: {bts(payload)}"
        )
    except Exception as e:
        return f"Failed to decode payload: {e}"
def decode_mqtt(payload):
    """
    MQTT 패킷 디코딩 함수 (기본 CONNECT 메시지 기반)
    """
    try:
        if payload[0] != 0x10:
            return "Not an MQTT CONNECT packet"

        protocol_name_len = int.from_bytes(payload[6:8], 'big')
        protocol_name = payload[8:8+protocol_name_len].decode(errors='ignore')
        protocol_name = bts(protocol_name.encode())
        try:
            protocol_level = payload[8+protocol_name_len]
        except IndexError:
            protocol_level = "Unknown"
        try:
            connect_flags = f"0x{payload[9+protocol_name_len]:02x}"
        except IndexError:
            connect_flags = "Unknown"
        keep_alive = int.from_bytes(payload[10+protocol_name_len:12+protocol_name_len], 'big')
        raw_data = bts(payload)
        result = (
            f"Detected Protocol: MQTT CONNECT Packet\n"
            f"Protocol: {protocol_name}\n"
            f"Level: {protocol_level}\n"
            f"Connect Flags: {connect_flags}\n"
            f"Keep Alive: {keep_alive} seconds\n"
            f"Readable ASCII Data: {raw_data}"
        )
        return result
    except Exception as e:
        return f"MQTT Decode Error: {str(e)}"

def decode_vnc(payload):
    try:
        version = payload[0:12].decode(errors="ignore").strip()
        security_types_count = payload[12]
        security_types = list(payload[13:13 + security_types_count])
        sec_names = {
            1: "None",
            2: "VNC Authentication",
            16: "Tight",
            18: "VeNCrypt",
            19: "GTK-VNC SASL",
            20: "RA2",
            21: "RA2ne",
            22: "SSPI",
            23: "MS-Logon",
        }
        decoded_ = []
        for s in security_types:
            if s not in sec_names:
                continue
            else:
                decoded_.append(sec_names[s])
        decode_data = bts(payload[13:])
        return (
            f"Detected Protocol: VNC\n"
            f"RFB Version: {version}\n"
            f"Supported Security Types ({security_types_count}): {', '.join(decoded_)}"
            f"Readable ASCII Data: {decode_data}"
        )
    except Exception as e:
        return f"Decoding Error: {str(e)}"


def decode_ws_discovery(payload):
    """
    WS-Discovery SOAP 메시지를 사람이 읽을 수 있도록 요약 디코드
    """
    try:
        xml_text = payload.decode("utf-8", errors="ignore")

        # 핵심 필드 추출 (간단한 예시)
        root = ET.fromstring(xml_text)

        ns = {
            'env': 'http://www.w3.org/2003/05/soap-envelope',
            'wsa': 'http://schemas.xmlsoap.org/ws/2004/08/addressing',
            'wsd': 'http://schemas.xmlsoap.org/ws/2005/04/discovery'
        }

        action = root.find(".//wsa:Action", ns)
        message_id = root.find(".//wsa:MessageID", ns)
        sequence = root.find(".//wsd:AppSequence", ns)

        result = (
            f"Detected Protocol: WS-Discovery SOAP Message\n"
            f"Action: {action.text if action is not None else 'N/A'}\n"
            f"MessageID: {message_id.text if message_id is not None else 'N/A'}\n"
            f"Sequence ID: {sequence.attrib.get('SequenceId') if sequence is not None else 'N/A'}\n"
            f"Message Number: {sequence.attrib.get('MessageNumber') if sequence is not None else 'N/A'}\n"
            f"Full XML:\n{xml_text}"
        )
        return result
    except Exception as e:
        return bytes_to_str_32126(payload)

def decode_ike(payload):
    """
    포트 500 IKE 패킷 디코더 (기본 구조만 추출)
    """
    try:
        initiator_spi = payload[:8].hex()
        responder_spi = payload[8:16].hex()
        next_payload = payload[16]
        version = payload[17]
        exchange_type = payload[18]
        flags = payload[19]
        message_id = int.from_bytes(payload[20:24], "big")
        length = int.from_bytes(payload[24:28], "big")
        ASCII_data = bts(payload[28:])
        result = (
            f"Detected Protocol: IKE (IPSec) Packet\n"
            f"Initiator SPI: {initiator_spi}\n"
            f"Responder SPI: {responder_spi}\n"
            f"Next Payload: 0x{next_payload:02x}\n"
            f"Version: 0x{version:02x}\n"
            f"Exchange Type: {exchange_type}\n"
            f"Flags: 0x{flags:02x}\n"
            f"Message ID: {message_id}\n"
            f"Total Length: {length} bytes\n"
            f"Readable ASCII Data: {ASCII_data}"
        )
        return result
    except Exception as e:
        return bytes_to_str_32126(payload, "IKE Decode Error: {e}")


def decode_coap(payload):
    try:
        version = (payload[0] & 0xC0) >> 6
        msg_type = (payload[0] & 0x30) >> 4
        token_len = payload[0] & 0x0F
        code = payload[1]
        msg_id = int.from_bytes(payload[2:4], 'big')

        uri_path = b''.join(payload[4:]).decode(errors='ignore')
        decoded = bts(payload[4:])
        result = (
            f"Detected Protocol: CoAP Packet\n"
            f"Version: {version}\n"
            f"Type: {msg_type} ({['Confirmable', 'Non-confirmable', 'Acknowledgement', 'Reset'][msg_type]})\n"
            f"Code: {code} ({'GET' if code == 1 else 'Other'})\n"
            f"Message ID: {msg_id}\n"
            f"URI-Path: {uri_path}"
            f"Readable ASCII Data: {decoded}"
        )
        return result
    except Exception as e:
        return f"[Decode Error] {e}"

def decode_rsync(payload):
    try:
        text = payload.decode(errors='ignore')
        parts = text.split('_')
        if len(parts) == 3:
            label, ip, port = parts
            return (
                f"Detected Protocol: Rsync Protocol Packet (Custom Format)\n"
                f"Label: {label}\n"
                f"Target IP: {ip}\n"
                f"Target Port: {port}"
                f"Readable ASCII Data: {text}"
            )
        return bytes_to_str_32126(payload, "rsync")
    except Exception as e:
        return bytes_to_str_32126(payload, f"rsync({e})")

def decode_custom_dvkt(payload):
    try:
        sig = payload[:4].decode()
        msg_type = int.from_bytes(payload[4:6], 'big')
        field = int.from_bytes(payload[6:8], 'big')
        mac = ':'.join(f'{b:02X}' for b in payload[8:14])
        try:
            decoded = bts(payload)
        except:
            decoded = ''
        return (
            f"Detected Protocol: Custom DVKT Protocol Packet\n"
            f"Signature: {sig}\n"
            f"Message Type: 0x{msg_type:04X}\n"
            f"Field Value: 0x{field:04X}\n"
            f"MAC Address: {mac}\n"
            f"Readable ASCII Data: {decoded}"
        )
    except Exception as e:
        return bytes_to_str_32126(payload, f"Custom DVKT({e})")


def decode_custom_domain_tunnel(payload):
    try:
        text = bts(payload)
        domains = [d for d in text.split() if '.' in d and len(d) > 5]
        return (
            f"Detected Protocol: Custom Domain Tunnel Packet\n"
            f"Domains Found: {', '.join(domains)}\n"
            f"Readable ASCII Data: : {text}"
        )
    except Exception as e:
        return bytes_to_str_32126(payload, f"Custom Domain Tunnel Packet({e})")

def decode_mgcp(payload):
    """
    MGCP 패킷을 사람이 읽을 수 있도록 디코딩하는 함수
    """
    try:
        text = payload.decode('utf-8', errors='ignore')
        lines = text.strip().split('\n')
        parsed = {}
        results = (
            f"Detected Protocol: mgcp\n"
        )
        # 첫 줄: 명령어, 트랜잭션 ID, 엔드포인트, 버전
        if lines:
            parts = lines[0].split()
            if len(parts) >= 4:
                parsed['Command'] = parts[0]
                parsed['TransactionID'] = parts[1]
                parsed['Endpoint'] = parts[2]
                parsed['Version'] = parts[3]

            for key, value in parsed.items():
                results += f"{key}: {value}\n"

        # 나머지 줄들: 키:값 형식
        for line in lines[1:]:
            if ':' in line:
                key, value = line.split(':', 1)
                results += f"{key.strip()}: {value.strip()}\n"

        results += f"Readable ASCII Data: {bts(payload)}"
        return results

    except Exception as e:
        return bytes_to_str_32126(payload, f"mgcp({e})")

def decode_tds_connectionless(payload):
    """
    TDS CONNECTIONLESS_TDS 메시지 디코딩
    """
    try:
        marker = b'CONNECTIONLESS_TDS'
        offset = payload.find(marker)
        results = (
            f"Detected Protocol: TDS\n"
            "mode: CONNECTIONLESS\n"
            f"marker_offset {offset}\n"
            f"marker {marker.decode()}\n"
            f"hex_preview {payload.hex()}\n"
            f"Readable ASCII Data: {bts(payload)}"

        )
        if offset == -1:
            return bytes_to_str_32126(payload, f"TDS(TDS marker not found)")

        return results
    except Exception as e:
        return bytes_to_str_32126(payload, f"TDS({e})")

def decode_printer(payload):
    """
    Canon 프린터용 BJNP 프로토콜 해석
    구조: 시그니처 'BJNP' + 버전 + 명령 ID + 데이터 블록 등
    """
    try:
        offset = 0
        results = (
            f"Detected Protocol: PRINTER\n"
        )
        while offset + 16 <= len(payload):
            segment = payload[offset:offset + 16]

            # BJNP
            if segment.startswith(b'BJNP'):
                version = segment[4]
                command = segment[5]
                results += (
                    f"BJNP offset {offset} | version: {version} | command: {command} | raw: {segment.hex()}\n"
                )
                offset += 16
                continue

            # EPSONP
            elif segment.startswith(b'EPSONP'):
                version = segment[6]
                command = segment[7]
                results += (
                    f"EPSONP offset {offset} | version: {version} | command: {command} | raw: {segment.hex()}\n"
                )
                offset += 16
                continue

            # MFNP
            elif segment.startswith(b'MFNP'):
                version = segment[4]
                command = segment[5]
                results += (
                    f"MFNP offset {offset} | version: {version} | command: {command} | raw: {segment.hex()}\n"
                )
                offset += 16
                continue
            elif segment.startswith(b'CANON'):
                subtype = segment[5:8]
                results += (
                    f"CANON offset {offset} | subtype: {subtype} | raw: {segment.hex()}\n"
                )
                offset += 16
                continue
            offset += 1
        decoded = bts(payload)
        results += f"Readable ASCII Data: {decoded}"
        return results
    except Exception as e:
        return {"error": str(e)}
