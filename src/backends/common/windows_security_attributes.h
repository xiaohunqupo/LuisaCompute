//
// Created by mike on 1/11/26.
//

#pragma once

#ifdef LUISA_PLATFORM_WINDOWS

#include <windows.h>
#include <VersionHelpers.h>
#include <dxgi1_2.h>
#include <AclAPI.h>

namespace luisa::compute {

class WindowsSecurityAttributes {

protected:
    SECURITY_ATTRIBUTES m_winSecurityAttributes{};
    PSECURITY_DESCRIPTOR m_winPSecurityDescriptor{};

public:
    WindowsSecurityAttributes() noexcept {
        m_winPSecurityDescriptor = (PSECURITY_DESCRIPTOR)calloc(
            1, SECURITY_DESCRIPTOR_MIN_LENGTH + 2 * sizeof(void **));
        PSID *ppSID = (PSID *)((PBYTE)m_winPSecurityDescriptor + SECURITY_DESCRIPTOR_MIN_LENGTH);
        PACL *ppACL = (PACL *)((PBYTE)ppSID + sizeof(PSID *));
        InitializeSecurityDescriptor(m_winPSecurityDescriptor, SECURITY_DESCRIPTOR_REVISION);
        SID_IDENTIFIER_AUTHORITY sidIdentifierAuthority = SECURITY_WORLD_SID_AUTHORITY;
        AllocateAndInitializeSid(&sidIdentifierAuthority, 1, SECURITY_WORLD_RID,
                                 0, 0, 0, 0, 0, 0, 0, ppSID);
        EXPLICIT_ACCESS explicitAccess;
        ZeroMemory(&explicitAccess, sizeof(EXPLICIT_ACCESS));
        explicitAccess.grfAccessPermissions = STANDARD_RIGHTS_ALL | SPECIFIC_RIGHTS_ALL;
        explicitAccess.grfAccessMode = SET_ACCESS;
        explicitAccess.grfInheritance = INHERIT_ONLY;
        explicitAccess.Trustee.TrusteeForm = TRUSTEE_IS_SID;
        explicitAccess.Trustee.TrusteeType = TRUSTEE_IS_WELL_KNOWN_GROUP;
        explicitAccess.Trustee.ptstrName = (LPTSTR)*ppSID;
        SetEntriesInAcl(1, &explicitAccess, nullptr, ppACL);
        SetSecurityDescriptorDacl(m_winPSecurityDescriptor, true, *ppACL, false);
        m_winSecurityAttributes.nLength = sizeof(m_winSecurityAttributes);
        m_winSecurityAttributes.lpSecurityDescriptor = m_winPSecurityDescriptor;
        m_winSecurityAttributes.bInheritHandle = true;
    }
    ~WindowsSecurityAttributes() noexcept {
        PSID *ppSID = (PSID *)((PBYTE)m_winPSecurityDescriptor + SECURITY_DESCRIPTOR_MIN_LENGTH);
        PACL *ppACL = (PACL *)((PBYTE)ppSID + sizeof(PSID *));
        if (*ppSID) { FreeSid(*ppSID); }
        if (*ppACL) { LocalFree(*ppACL); }
        free(m_winPSecurityDescriptor);
    }
    [[nodiscard]] auto get() const noexcept {
        return &m_winSecurityAttributes;
    }
};

}// namespace luisa::compute

#endif
